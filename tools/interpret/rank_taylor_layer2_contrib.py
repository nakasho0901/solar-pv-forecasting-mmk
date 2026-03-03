# -*- coding: utf-8 -*-
r"""
rank_taylor_layer2_contrib.py

目的：
- 学習済み ckpt をロードして MMK_FusionPV_FeatureToken_v2 を復元
- test_x.npy をモデルに流す
- Layer2 の Taylor expert（例：expert2）に入る「トークン入力 x (64次元)」をフックで取得
- TaylorKAN の係数 coeffs[o,i,k] を使い、4096本の基底関数
    phi_{o,i}(x_i) = sum_{k=0..3} c[o,i,k] * (x_i^k)
  の “寄与度” を
    contrib[o,i] = mean_{test} |phi_{o,i}(x_i)|
  で計算（データ依存の寄与度）
- 寄与度上位の (o,i) を抽出し、数式（係数入り）と % を保存

寄与度の意味（このスクリプトの定義）：
- 「Layer2・Taylor expert 内で、テストデータに対して各基底関数が平均的にどれくらい大きな値を出していたか」
- ※ヘッド重み・他expert混合まで含む“最終予測寄与”ではない（それは別途拡張可能）

実行例（Windows ワンライナー）：
set PYTHONPATH=C:\Users\nakas\my_project\solar-kan && python rank_taylor_layer2_contrib.py --ckpt "C:\Users\nakas\my_project\solar-kan\save\MMK_FusionPV_FeatureToken\version_12\checkpoints\last.ckpt" --model_py "C:\Users\nakas\my_project\solar-kan\easytsf\model\MMK_FusionPV_FeatureToken_v2.py" --config_py "C:\Users\nakas\my_project\solar-kan\config\tsukuba_conf\MMK_FusionPV_FeatureToken_A11_stride1_v2.py" --test_x "C:\Users\nakas\my_project\solar-kan\dataset_PV\prepared\96-96_fusionA11_96_96_stride1\test_x.npy" --layer_index 2 --expert_id 2 --batch_size 256 --outdir "C:\Users\nakas\my_project\solar-kan\results_taylor_contrib_layer2"

出力：
- outdir/taylor_layer2_contrib_top.csv
- outdir/taylor_layer2_contrib_full.npz （contrib行列など）
- outdir/taylor_layer2_equations_top.txt （上位の数式）
"""

import os
import csv
import json
import argparse
import importlib.util
from typing import Any, Dict, Tuple

import numpy as np
import torch

# プロジェクト内 import（PYTHONPATHが通っている前提）
from easytsf.layer.kanlayer import TaylorKANLayer


# -------------------------
# util: ファイルから module をロード
# -------------------------
def load_module_from_path(mod_name: str, path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"module file not found: {path}")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -------------------------
# util: ckpt から state_dict を取り出す（Lightning対応）
# -------------------------
def extract_state_dict(ckpt_obj: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        return ckpt_obj["state_dict"]
    if isinstance(ckpt_obj, dict) and all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
        return ckpt_obj
    raise ValueError("Unsupported checkpoint format: cannot find state_dict.")


def strip_prefix_if_needed(state_dict: Dict[str, torch.Tensor], prefixes=("model.", "net.", "module.")) -> Dict[str, torch.Tensor]:
    keys = list(state_dict.keys())
    for p in prefixes:
        if any(k.startswith(p) for k in keys):
            return {k[len(p):] if k.startswith(p) else k: v for k, v in state_dict.items()}
    return state_dict


# -------------------------
# core: phi_{o,i}(x_i) を一括計算して contrib を足し込む
# -------------------------
@torch.no_grad()
def accumulate_contrib(
    x: torch.Tensor,
    coeffs: torch.Tensor,
    sum_abs: torch.Tensor,
    n_seen: int,
) -> int:
    """
    x:      (B, 64)   Layer2 TaylorKAN の入力トークン
    coeffs: (64,64,4) TaylorKAN の係数 (out, in, power)
    sum_abs:(64,64)   |phi| の合計を蓄積
    n_seen: 既に処理したサンプル数

    return: 更新後の n_seen
    """
    # x のべき乗 (B, 64, 4) -> [x^0, x^1, x^2, x^3]
    # x^0 は 1
    B, I = x.shape
    x0 = torch.ones((B, I), device=x.device, dtype=x.dtype)
    x1 = x
    x2 = x * x
    x3 = x2 * x
    xpow = torch.stack([x0, x1, x2, x3], dim=-1)  # (B, I, 4)

    # poly[b,o,i] = sum_k coeffs[o,i,k] * xpow[b,i,k]
    # einsum: (B,I,K) and (O,I,K) -> (B,O,I)
    poly = torch.einsum("bik,oik->boi", xpow, coeffs)

    # |poly| を合計
    sum_abs += poly.abs().sum(dim=0)  # (O,I)
    n_seen += B
    return n_seen


def format_equation(c: np.ndarray, o: int, i: int) -> str:
    """
    c: (4,) coefficients [c0,c1,c2,c3]
    """
    c0, c1, c2, c3 = c.tolist()
    # 数式表示は小数を整形（丸めは表示だけ。CSVには生値も出す）
    return (
        f"phi_({o},{i})(x) = "
        f"{c0:+.6g} {c1:+.6g}*x {c2:+.6g}*x^2 {c3:+.6g}*x^3"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_py", required=True)
    ap.add_argument("--config_py", required=True)
    ap.add_argument("--test_x", required=True)
    ap.add_argument("--layer_index", type=int, default=2)
    ap.add_argument("--expert_id", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- config / model 読み込み
    conf_mod = load_module_from_path("exp_conf_mod", args.config_py)
    exp_conf = conf_mod.exp_conf

    model_mod = load_module_from_path("model_mod", args.model_py)
    ModelClass = getattr(model_mod, exp_conf["model_class"])

    model = ModelClass(
        hist_len=int(exp_conf["hist_len"]),
        pred_len=int(exp_conf["pred_len"]),
        var_num=int(exp_conf["var_num"]),
        pv_index=int(exp_conf.get("pv_index", 0)),
        token_dim=int(exp_conf["token_dim"]),
        layer_num=int(exp_conf["layer_num"]),
        layer_hp=exp_conf["layer_hp"],
        use_layernorm=bool(exp_conf["use_layernorm"]),
        dropout=float(exp_conf["dropout"]),
        fusion=str(exp_conf["fusion"]),
        baseline_mode=str(exp_conf["baseline_mode"]),
        use_pv_revin=bool(exp_conf["use_pv_revin"]),
        revin_eps=float(exp_conf["revin_eps"]),
        enforce_nonneg=bool(exp_conf["enforce_nonneg"]),
    )

    # --- ckpt ロード
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = strip_prefix_if_needed(extract_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # --- device
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.to(device)
    model.eval()

    # --- 対象の TaylorKANLayer を特定
    try:
        blk = model.blocks[args.layer_index]
        mok = blk.mok
        ex = mok.experts[args.expert_id]
        tr = getattr(ex, "transform", None)
    except Exception as e:
        raise RuntimeError(f"Failed to locate layer/block/expert: {e}")

    if tr is None or not isinstance(tr, TaylorKANLayer):
        raise RuntimeError(
            f"Selected expert is not TaylorKANLayer. layer_index={args.layer_index} expert_id={args.expert_id} "
            f"(transform type={type(tr)})"
        )

    coeffs = tr.coeffs.detach().to(device).float()  # (64,64,4)

    # --- test_x 読み込み
    X = np.load(args.test_x)
    # 想定: (N, hist_len, var_num) もしくは (N, hist_len, feat_dim) 等
    # モデル forward にそのまま渡せる形にする
    X_t = torch.from_numpy(X).to(device)
    if X_t.dtype != torch.float32:
        X_t = X_t.float()

    N = X_t.shape[0]

    # --- Layer2 TaylorKAN の入力トークンをフックで受け取る
    #     ここで「実際にモデルがTaylorKANに入れる x」を得る
    hook_cache: Dict[str, torch.Tensor] = {}

    def taylor_input_hook(module, inp, out):
        # inp は tuple。最初が入力テンソルであることが多い
        x_in = inp[0]
        # 形状が (B,64) になることを期待。もし (B,*,64) なら平均して (B,64) に落とす
        if x_in.ndim == 3:
            # 例: (B, T, 64) -> 時間次元を平均（最終層入力としての代表）
            x_in = x_in.mean(dim=1)
        if x_in.ndim != 2 or x_in.shape[1] != 64:
            raise RuntimeError(f"Unexpected Taylor input shape: {tuple(x_in.shape)} (expected (B,64))")
        hook_cache["x_in"] = x_in.detach()

    h = tr.register_forward_hook(taylor_input_hook)

    # --- 寄与度を蓄積
    sum_abs = torch.zeros((64, 64), device=device, dtype=torch.float32)
    n_seen = 0

    # forward するだけの関数（モデルが返すものは使わない）
    # 多くのモデルは forward(x) を想定しているが、違う場合はここを修正する
    def run_forward(batch_x: torch.Tensor):
        # 返り値は不要、フックで x_in を受け取る
        _ = model(batch_x)

    # --- バッチ処理
    bs = int(args.batch_size)
    for s in range(0, N, bs):
        e = min(N, s + bs)
        bx = X_t[s:e]

        hook_cache.clear()
        run_forward(bx)

        if "x_in" not in hook_cache:
            raise RuntimeError("Taylor input hook did not capture input. Forward path may differ.")

        x_in = hook_cache["x_in"].float()  # (B,64)
        n_seen = accumulate_contrib(x_in, coeffs, sum_abs, n_seen)

    # hook解除
    h.remove()

    # --- 平均寄与度（mean abs）
    contrib = (sum_abs / max(1, n_seen)).detach().cpu().numpy()  # (64,64)
    total = float(contrib.sum())
    percent = (contrib / total * 100.0) if total > 0 else np.zeros_like(contrib)

    # --- 上位topkを抽出
    flat_idx = np.argsort(-contrib.reshape(-1))
    topk = min(int(args.topk), flat_idx.size)

    # 係数もCPUへ
    coeffs_cpu = coeffs.detach().cpu().numpy()  # (64,64,4)

    # --- CSV保存
    csv_path = os.path.join(args.outdir, "taylor_layer2_contrib_top.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "out_dim_o", "in_dim_i", "mean_abs_contrib", "percent_of_total", "c0", "c1", "c2", "c3"])
        for r in range(topk):
            idx = int(flat_idx[r])
            o = idx // 64
            i = idx % 64
            c = coeffs_cpu[o, i, :]
            w.writerow([
                r + 1, o, i,
                float(contrib[o, i]),
                float(percent[o, i]),
                float(c[0]), float(c[1]), float(c[2]), float(c[3]),
            ])

    # --- 上位の数式テキスト保存
    txt_path = os.path.join(args.outdir, "taylor_layer2_equations_top.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"[INFO] layer_index={args.layer_index}, expert_id={args.expert_id}\n")
        f.write(f"[INFO] n_seen={n_seen}, total_mean_abs={total:.6g}\n\n")
        for r in range(topk):
            idx = int(flat_idx[r])
            o = idx // 64
            i = idx % 64
            c = coeffs_cpu[o, i, :]
            eq = format_equation(c, o, i)
            f.write(f"rank {r+1:>3d} | percent={percent[o,i]:8.4f}% | mean_abs={contrib[o,i]:.6g} | {eq}\n")

    # --- 全体行列も保存（後で可視化・ヒートマップ用）
    npz_path = os.path.join(args.outdir, "taylor_layer2_contrib_full.npz")
    np.savez_compressed(
        npz_path,
        contrib=contrib,
        percent=percent,
        coeffs=coeffs_cpu,
        n_seen=np.array([n_seen], dtype=np.int64),
        missing=np.array(missing, dtype=object),
        unexpected=np.array(unexpected, dtype=object),
        layer_index=np.array([args.layer_index], dtype=np.int64),
        expert_id=np.array([args.expert_id], dtype=np.int64),
    )

    # --- 実行ログ
    print("[OK] saved:", args.outdir)
    print(" -", csv_path)
    print(" -", txt_path)
    print(" -", npz_path)
    print("[load_state_dict] missing:", len(missing), "unexpected:", len(unexpected))
    print("[summary] n_seen:", n_seen, "top1_percent:", float(percent.reshape(-1)[flat_idx[0]]))


if __name__ == "__main__":
    main()
