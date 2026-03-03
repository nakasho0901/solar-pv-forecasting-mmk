# -*- coding: utf-8 -*-
"""
最後の窓（testデータの最後サンプル）について、
「96時間（予測ステップ）× expert」の寄与ヒートマップを作る。

重要ポイント：
- MMK_Mix の gate(score) は (B*N, E) で、出力の96次元ごとには変化しない設計。
  そのため「96時間ごとの expert 切替」を直接 gate だけで見ることはできない。
- 代わりに、各 expert の出力ベクトル（長さ96）と gate を用いて
  各ステップの寄与度を定義する：
    contrib(step, expert) = |expert_output(step)| * gate(expert)
  を expert 方向に正規化して、96×expert の比率として可視化する。

出力：
- results_last_window/
  - last_window_layer2_pv_96xexpert.png
  - last_window_layer2_varmean_96xexpert.png
  - last_window_meta.txt

実行例（CMDワンライナー）：
  python tools\\extract_last_window_96xexpert_heatmap.py --model_dir save_mmk_optimized --config config\\tsukuba_conf\\MMK_Mix_PV_96.py --out_dir results_last_window
"""

import os
import sys
import glob
import argparse
import importlib.util

import numpy as np
import torch
import matplotlib.pyplot as plt


# -------------------------
# パス設定（tools配下から親ディレクトリを import 可能にする）
# -------------------------
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(current_script_dir)  # solar-kan
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# あなたの既存コード資産を流用（compare_models_kw.py / final_visual_showdown.py と同系列）
from traingraph import PeakRunner, get_model  # noqa


def find_checkpoint(model_dir: str) -> str:
    """model_dir 配下から ckpt を見つける（best優先→なければ最新）"""
    search_pattern = os.path.join(project_dir, model_dir, "**", "*.ckpt")
    ckpts = glob.glob(search_pattern, recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"[ERROR] {model_dir} 配下に .ckpt が見つかりません: {search_pattern}")

    best = [p for p in ckpts if "best" in os.path.basename(p).lower()]
    target = max(best, key=os.path.getmtime) if best else max(ckpts, key=os.path.getmtime)
    return target


def load_config(config_path: str) -> dict:
    """config/*.py の exp_conf を読み込む"""
    abs_path = os.path.join(project_dir, config_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"[ERROR] config が見つかりません: {abs_path}")

    spec = importlib.util.spec_from_file_location("config_module", abs_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)  # type: ignore
    if not hasattr(config_module, "exp_conf"):
        raise AttributeError(f"[ERROR] {config_path} に exp_conf がありません")
    return config_module.exp_conf


def plot_96xexpert_heatmap(weights_96xE: np.ndarray, expert_names, save_path: str, title: str):
    """
    weights_96xE: shape (96, E)
    """
    E = weights_96xE.shape[1]
    plt.figure(figsize=(14, 4))

    # Expert を縦軸にしたいので (E, 96) を描画
    img = plt.imshow(weights_96xE.T, aspect="auto", interpolation="nearest")
    plt.colorbar(img, label="Normalized contribution [-]")

    if expert_names and len(expert_names) == E:
        plt.yticks(np.arange(E), expert_names)
    else:
        plt.yticks(np.arange(E), [f"expert{i}" for i in range(E)])

    # 予測ステップ（1..96）
    plt.xlabel("Forecast step (t+1 ... t+96) [-]")
    plt.ylabel("Expert [-]")

    # 目盛りを読みやすく
    xticks = [0, 11, 23, 35, 47, 59, 71, 83, 95]
    xlabels = ["1", "12", "24", "36", "48", "60", "72", "84", "96"]
    plt.xticks(xticks, xlabels)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="save_mmk_optimized", help="ckpt が入っているディレクトリ（project_dirからの相対）")
    ap.add_argument("--config", default=r"config\tsukuba_conf\MMK_Mix_PV_96.py", help="exp_conf を含む config")
    ap.add_argument("--out_dir", default="results_last_window", help="出力先フォルダ")
    ap.add_argument("--device", default="cpu", help="cpu または cuda（使える場合）")
    ap.add_argument("--use_ghi_boost", action="store_true", help="学習時と合わせて GHI を 1.3倍にする（既存コード互換）")
    ap.add_argument("--pv_var_index", type=int, default=0, help="PVの変数インデックス（デフォルト0）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # -------------------------
    # ckpt / config / dataloader
    # -------------------------
    ckpt_path = find_checkpoint(args.model_dir)
    exp_conf = load_config(args.config)

    # NPYDataInterface はあなたの環境にある前提（train.py のログと一致）
    from easytsf.runner.data_runner import NPYDataInterface  # noqa
    data_module = NPYDataInterface(**exp_conf)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    # モデルロード
    base_model = get_model(exp_conf["model_name"], exp_conf)
    runner = PeakRunner.load_from_checkpoint(ckpt_path, model=base_model, config=exp_conf, strict=False).eval()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    runner.to(device)

    # expert名
    expert_names = None
    if isinstance(exp_conf.get("layer_hp", None), (list, tuple)):
        expert_names = [str(x[0]) if isinstance(x, (list, tuple)) and len(x) > 0 else str(x) for x in exp_conf["layer_hp"]]

    # -------------------------
    # test の「最後サンプル」を取り出す
    # -------------------------
    last_batch = None
    for batch in test_loader:
        last_batch = batch

    if last_batch is None:
        raise RuntimeError("[ERROR] test_loader が空です。")

    # batch 想定：var_x, marker_x, var_y, marker_y
    var_x, marker_x, var_y, marker_y = last_batch
    var_x = var_x.float()
    marker_x = marker_x.float()

    # 最後のサンプル（Bの最後）
    var_x_last = var_x[-1:].to(device)       # (1, L, N)
    marker_x_last = marker_x[-1:].to(device) # (1, L, ?)

    # 学習条件に寄せる（必要なら）
    if args.use_ghi_boost:
        # 既存の比較コードが var_x[:,:,2] をGHIとして 1.3倍にしていた流れに合わせる
        var_x_last = var_x_last.clone()
        var_x_last[:, :, 2] *= 1.3

    # -------------------------
    # MoKLayer を探す（3層想定）
    # ここでは「最後の層（layer2）」だけを 96ステップ解釈したい
    # -------------------------
    mok_layers = [m for m in runner.modules() if m.__class__.__name__ == "MoKLayer"]
    if len(mok_layers) == 0:
        raise RuntimeError("[ERROR] MoKLayer が見つかりません。MMK_Mix の読み込みに失敗している可能性があります。")

    # layer2（最後のMoKLayer）を対象にする
    target_layer = mok_layers[-1]
    target_layer_index = len(mok_layers) - 1

    # -------------------------
    # forward hook で「入力x」を取り、expert出力をその場で再計算して保存
    # （モデル本体を改造しないための方法）
    # -------------------------
    cache = {}

    def hook_fn(module, inputs, outputs):
        # inputs[0] = x: (B*N, in_features)
        x_in = inputs[0]  # Tensor
        # outputs = (y, score)
        if not (isinstance(outputs, (tuple, list)) and len(outputs) >= 2):
            return
        score = outputs[1]  # (B*N, E)

        # expert_outputs を再計算： (B*N, out_features, E)
        with torch.no_grad():
            exp_outs = torch.stack([exp(x_in) for exp in module.experts], dim=-1)

        cache["x_in"] = x_in.detach().cpu()
        cache["score"] = score.detach().cpu()
        cache["expert_outputs"] = exp_outs.detach().cpu()

    h = target_layer.register_forward_hook(hook_fn)

    # 推論（1サンプルだけ）
    with torch.no_grad():
        _ = runner(var_x_last, marker_x_last)

    # hook解除
    h.remove()

    if "expert_outputs" not in cache or "score" not in cache:
        raise RuntimeError("[ERROR] hook で gate / expert_outputs を取得できませんでした。")

    score = cache["score"].numpy()  # (B*N, E)
    exp_outs = cache["expert_outputs"].numpy()  # (B*N, out_features, E)

    # -------------------------
    # 形状を (B=1, N, out_features, E) に整形
    # last window は B=1 なので、B_total=1
    # -------------------------
    var_num = int(exp_conf.get("var_num", var_x_last.shape[2]))
    E = score.shape[1]
    out_features = exp_outs.shape[1]

    # ここで out_features が 96（pred_len）であることを期待
    # もし違う場合、layer2 が pred_len を出していない = モデル設計が想定と違う
    if out_features != int(exp_conf.get("pred_len", 96)):
        # それでも可視化はできるが、「96時間」の意味が崩れるので警告して止める
        raise RuntimeError(f"[ERROR] target layer の out_features={out_features} が pred_len(={exp_conf.get('pred_len', 96)}) と一致しません。"
                           f" layerの選択が違う可能性があります。")

    # (B*N, out_features, E) -> (B, N, out_features, E)
    exp_outs_bn = exp_outs.reshape(1, var_num, out_features, E)
    score_bn = score.reshape(1, var_num, E)

    # -------------------------
    # 「96×expert」寄与度を定義して可視化
    # contrib(step, expert) = |expert_output(step)| * gate(expert)
    # これを expert 方向で正規化して比率にする
    # -------------------------
    eps = 1e-12

    # (1, N, 96, E)
    contrib = np.abs(exp_outs_bn) * score_bn[:, :, None, :]

    # 1) PV変数だけ（pv_var_index）
    pv_i = int(args.pv_var_index)
    if pv_i < 0 or pv_i >= var_num:
        raise ValueError(f"[ERROR] pv_var_index={pv_i} が不正です。var_num={var_num}")

    contrib_pv = contrib[0, pv_i, :, :]  # (96, E)
    denom_pv = contrib_pv.sum(axis=1, keepdims=True) + eps
    w_pv = contrib_pv / denom_pv  # (96, E)

    # 2) 変数平均（var_mean）
    contrib_mean = contrib[0].mean(axis=0)  # (96, E)
    denom_mean = contrib_mean.sum(axis=1, keepdims=True) + eps
    w_mean = contrib_mean / denom_mean  # (96, E)

    # 保存
    np.save(os.path.join(args.out_dir, "last_window_layer2_pv_96xexpert.npy"), w_pv)
    np.save(os.path.join(args.out_dir, "last_window_layer2_varmean_96xexpert.npy"), w_mean)

    plot_96xexpert_heatmap(
        w_pv,
        expert_names,
        os.path.join(args.out_dir, "last_window_layer2_pv_96xexpert.png"),
        title=f"Last window: layer{target_layer_index} (PV var {pv_i}) 96xexpert contribution"
    )

    plot_96xexpert_heatmap(
        w_mean,
        expert_names,
        os.path.join(args.out_dir, "last_window_layer2_varmean_96xexpert.png"),
        title=f"Last window: layer{target_layer_index} (var-mean) 96xexpert contribution"
    )

    # メタ情報
    meta_lines = [
        f"ckpt_path={ckpt_path}",
        f"config={args.config}",
        f"target_layer_index={target_layer_index} (last MoKLayer)",
        f"var_num={var_num}",
        f"pred_len={exp_conf.get('pred_len', 96)}",
        f"expert_names={expert_names}",
        f"use_ghi_boost={args.use_ghi_boost}",
        "definition: contrib(step,expert)=|expert_output(step)|*gate(expert), normalized over experts per step",
    ]
    with open(os.path.join(args.out_dir, "last_window_meta.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(meta_lines))

    print("[SUCCESS] 最後の窓について 96×expert ヒートマップを保存しました。")
    print(f"  out_dir: {os.path.abspath(args.out_dir)}")
    print("  - last_window_layer2_pv_96xexpert.png")
    print("  - last_window_layer2_varmean_96xexpert.png")


if __name__ == "__main__":
    main()
