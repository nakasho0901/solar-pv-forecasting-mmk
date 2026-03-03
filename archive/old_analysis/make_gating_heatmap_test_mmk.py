# tools/make_gating_heatmap_test_mmk.py
# -*- coding: utf-8 -*-
r"""
MMK_FusionPV_FeatureToken: testデータでゲーティング(Expert選択確率)を集計し、
層ごとのヒートマップ(PNG) + 行列(CSV/NPZ)を保存する。

前提:
- model(x, return_gate=True) -> (pred, gates_all)
- gates_all は list で、各要素 gate_tensor の形状はだいたい (B, F, E)
  (B=batch, F=features(var), E=experts)
"""

import os
import json
import argparse
import importlib.util
from typing import List, Tuple, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Config読み込み
# ---------------------------------------------------------
def load_exp_conf(config_path: str) -> dict:
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, "exp_conf"):
        raise AttributeError("Config must define `exp_conf` dict.")
    return module.exp_conf


# ---------------------------------------------------------
# meta.json から特徴量名を取得
# ---------------------------------------------------------
def load_feature_names(meta_json_path: str, fallback_num: int) -> List[str]:
    if os.path.exists(meta_json_path):
        with open(meta_json_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cols = meta.get("columns", {})
        names = cols.get("feature_cols", [])
        if isinstance(names, list) and len(names) > 0:
            return names
    return [f"Feat_{i}" for i in range(fallback_num)]


# ---------------------------------------------------------
# lightning ckpt から state_dict を取り出す
# ---------------------------------------------------------
def load_state_dict_from_ckpt(ckpt_path: str, map_location: str = "cpu") -> dict:
    ckpt = torch.load(ckpt_path, map_location=map_location)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("ckpt must be a lightning checkpoint dict with key 'state_dict'.")
    sd = ckpt["state_dict"]
    # Lightning で 'model.' が付いている場合があるので除去
    new_sd = {}
    for k, v in sd.items():
        nk = k[len("model."):] if k.startswith("model.") else k
        new_sd[nk] = v
    return new_sd


# ---------------------------------------------------------
# model import（プロジェクト側の easytsf を使う）
# ---------------------------------------------------------
def build_model_from_conf(exp_conf: dict):
    """
    ここはプロジェクト構成に依存するので、基本は easytsf 側の import に任せる。
    Geminiコード同様、まず標準 import を試し、失敗時は file からの import にフォールバック。
    """
    # まずは通常 import（推奨）
    try:
        from easytsf.model.MMK_FusionPV_FeatureToken import MMK_FusionPV_FeatureToken
        ModelClass = MMK_FusionPV_FeatureToken
    except Exception:
        # フォールバック：model_module が書いてある前提で import
        model_module = exp_conf.get("model_module", "")
        if not model_module:
            raise ImportError("Cannot import MMK_FusionPV_FeatureToken and exp_conf has no model_module.")
        # 例: easytsf.model.MMK_FusionPV_FeatureToken_v2 など
        mod = __import__(model_module, fromlist=["*"])
        model_class_name = exp_conf.get("model_class", "MMK_FusionPV_FeatureToken")
        if not hasattr(mod, model_class_name):
            raise ImportError(f"Module {model_module} has no class {model_class_name}")
        ModelClass = getattr(mod, model_class_name)

    # conf から必要そうな引数を渡して初期化
    model = ModelClass(
        hist_len=exp_conf["hist_len"],
        pred_len=exp_conf["pred_len"],
        var_num=exp_conf["var_num"],
        pv_index=exp_conf.get("pv_index", 0),
        token_dim=exp_conf.get("token_dim", 64),
        layer_num=exp_conf.get("layer_num", 3),
        layer_hp=exp_conf.get("layer_hp"),
        use_layernorm=exp_conf.get("use_layernorm", True),
        dropout=exp_conf.get("dropout", 0.1),
        fusion=exp_conf.get("fusion", "mean"),
        baseline_mode=exp_conf.get("baseline_mode", "last24_repeat"),
        use_pv_revin=exp_conf.get("use_pv_revin", True),
        revin_eps=exp_conf.get("revin_eps", 1e-5),
        enforce_nonneg=exp_conf.get("enforce_nonneg", True),
    )
    return model


# ---------------------------------------------------------
# gate を (F,E) に集計（全test平均）
# ---------------------------------------------------------
def accumulate_gate_means(
    model,
    x_np: np.ndarray,
    device: torch.device,
    batch_size: int,
    max_batches: int = -1,
    use_day_mask: bool = False,
    marker_y_np: Optional[np.ndarray] = None,
    day_threshold: float = 0.5,
) -> Tuple[List[np.ndarray], List[int]]:
    """
    戻り値:
      gate_sum_list: 各層の (F,E) 合計
      gate_count_list: 各層で足し込んだサンプル数（重み）
    """
    model.eval()

    n = x_np.shape[0]
    gate_sum_list = None
    gate_count_list = None

    with torch.no_grad():
        for bi, start in enumerate(range(0, n, batch_size)):
            end = min(n, start + batch_size)
            xb = torch.from_numpy(x_np[start:end]).float().to(device)

            # (pred, gates_all)
            out = model(xb, return_gate=True)
            if not isinstance(out, (tuple, list)) or len(out) < 2:
                raise RuntimeError("model(x, return_gate=True) must return (pred, gates_all).")
            gates_all = out[1]  # list of tensors

            # 初回だけ器を作る
            if gate_sum_list is None:
                gate_sum_list = []
                gate_count_list = []
                for g in gates_all:
                    # g: (B,F,E) を想定
                    if g.ndim != 3:
                        raise ValueError(f"Expected gate tensor ndim=3 (B,F,E). Got shape={tuple(g.shape)}")
                    F, E = int(g.shape[1]), int(g.shape[2])
                    gate_sum_list.append(np.zeros((F, E), dtype=np.float64))
                    gate_count_list.append(0)

            # day mask で重み付け平均する場合（任意）
            # marker_y は予測ホライズン側の daylight(0/1) なので、
            # 「その窓が昼を含むか」を window weight として使う（簡易）
            if use_day_mask:
                if marker_y_np is None:
                    raise ValueError("use_day_mask=True requires marker_y_np.")
                my = marker_y_np[start:end]  # (B, pred_len) を想定
                if my.ndim != 2:
                    raise ValueError(f"marker_y expected (B, pred_len). Got {my.shape}")
                # 窓ごとの昼割合
                day_ratio = my.mean(axis=1)  # (B,)
                # 閾値以上の窓だけ採用（昼がほぼ無い窓を落とす）
                keep = day_ratio >= day_threshold
                if keep.sum() == 0:
                    continue
                keep_t = torch.from_numpy(keep).to(device)
            else:
                keep_t = None

            # 各層を足し込み
            for li, g in enumerate(gates_all):
                # g: (B,F,E)
                if keep_t is not None:
                    g_use = g[keep_t]
                else:
                    g_use = g
                if g_use.numel() == 0:
                    continue

                # batch 平均 -> (F,E)
                g_mean = g_use.mean(dim=0).detach().cpu().numpy().astype(np.float64)
                gate_sum_list[li] += g_mean
                gate_count_list[li] += 1  # “バッチ平均”を何回足したか

            if max_batches > 0 and (bi + 1) >= max_batches:
                break

    assert gate_sum_list is not None and gate_count_list is not None
    return gate_sum_list, gate_count_list


# ---------------------------------------------------------
# 描画 & 保存
# ---------------------------------------------------------
def save_heatmap(
    mat_FE: np.ndarray,  # (F,E)
    feature_names: List[str],
    expert_names: List[str],
    title: str,
    out_png: str,
):
    F, E = mat_FE.shape
    # 表示は (E,F) にすると “Experts x Features” で読みやすい
    data = mat_FE.T  # (E,F)

    # サイズ調整（Fが多いと縦に長くする）
    fig_w = max(10, F * 0.8)
    fig_h = max(4, E * 0.9)

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(111)
    im = ax.imshow(data, aspect="auto", vmin=0.0, vmax=1.0)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)

    ax.set_xlabel("Features")
    ax.set_ylabel("Experts")

    ax.set_xticks(list(range(F)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right")

    ax.set_yticks(list(range(E)))
    ax.set_yticklabels(expert_names, rotation=0)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def save_csv(mat_FE: np.ndarray, feature_names: List[str], expert_names: List[str], out_csv: str):
    # (F,E) -> CSV は (E,F) で保存（行=expert, 列=feature）
    data = mat_FE.T
    header = "," + ",".join(feature_names)
    lines = [header]
    for ei, en in enumerate(expert_names):
        row = [en] + [f"{data[ei, fj]:.6f}" for fj in range(data.shape[1])]
        lines.append(",".join(row))
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_x", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--max_batches", type=int, default=-1, help="debug用。-1で全batch")
    # オプション：夜窓を減らしたいとき（marker_yでフィルタ）
    ap.add_argument("--use_day_mask", action="store_true")
    ap.add_argument("--marker_y", default="")
    ap.add_argument("--day_threshold", type=float, default=0.5, help="窓の昼割合がこれ以上なら採用")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # load config/meta
    exp_conf = load_exp_conf(args.config)
    x_np = np.load(args.test_x)
    var_num_real = x_np.shape[2]
    exp_conf["var_num"] = var_num_real  # 念のため上書き

    feature_names = load_feature_names(args.meta_json, fallback_num=var_num_real)

    # expert名（あなたのlayer_hpは4expert想定）
    expert_names = ["KAN (Spline)", "WavKAN", "TaylorKAN", "JacobiKAN"]

    # device
    use_cuda = (args.device == "cuda" and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"[INFO] device={device}")

    # build model & load weights
    model = build_model_from_conf(exp_conf)
    sd = load_state_dict_from_ckpt(args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[INFO] load_state_dict strict=False | missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device)

    # optional marker_y
    marker_y_np = None
    if args.use_day_mask:
        if not args.marker_y:
            raise ValueError("--use_day_mask requires --marker_y path.")
        marker_y_np = np.load(args.marker_y)

    # accumulate
    gate_sum_list, gate_count_list = accumulate_gate_means(
        model=model,
        x_np=x_np,
        device=device,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        use_day_mask=args.use_day_mask,
        marker_y_np=marker_y_np,
        day_threshold=args.day_threshold,
    )

    # save per layer
    for li, (gs, gc) in enumerate(zip(gate_sum_list, gate_count_list)):
        if gc == 0:
            print(f"[WARN] layer{li}: no samples accumulated (gc=0). skipped.")
            continue

        gate_FE = gs / float(gc)  # (F,E)

        # save npz
        npz_path = os.path.join(args.outdir, f"layer{li}_gate_matrix_FE.npz")
        np.savez(
            npz_path,
            gate_FE=gate_FE.astype(np.float32),
            feature_names=np.array(feature_names, dtype=object),
            expert_names=np.array(expert_names, dtype=object),
        )
        print(f"[OK] saved: {npz_path} shape={gate_FE.shape}")

        # save csv
        csv_path = os.path.join(args.outdir, f"layer{li}_gate_matrix_expert_x_feature.csv")
        save_csv(gate_FE, feature_names, expert_names, csv_path)
        print(f"[OK] saved: {csv_path}")

        # save heatmap
        png_path = os.path.join(args.outdir, f"layer{li}_gate_heatmap.png")
        title = f"MMK gating heatmap (test mean) | layer {li} | shape(F,E)={gate_FE.shape}"
        save_heatmap(gate_FE, feature_names, expert_names, title, png_path)
        print(f"[OK] saved: {png_path}")

    print("[DONE] All outputs saved in:", args.outdir)


if __name__ == "__main__":
    main()
