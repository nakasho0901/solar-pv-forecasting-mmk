# -*- coding: utf-8 -*-
"""
MMK_Mix 24h: 最後の窓（testの最後サンプル）を kW でプロットする（compare_models_kw と同じ換算）

- 予測: pred[:,:,3] * KW_FACTOR
- 真値: var_y[:,:,0] * KW_FACTOR
- 学習時と同じ GHI boost を適用: var_x[:,:,2] *= 1.3

出力:
  results_24h_plot/
    - mmk_24h_last_window_kw.png
    - mmk_24h_last_window_kw.csv

実行（CMDワンライナー）:
  python tools\\plot_mmk_24h_last_window_kw.py
"""

import os
import sys
import csv
import numpy as np
import torch
import importlib.util
import matplotlib.pyplot as plt

# ===== 物理量変換（compare_models_kw.py と一致）=====
KW_FACTOR = 33.01  # [-] これを掛けると kW になる（あなたの環境の定義）
# OFFSET = 20.54  # compare_models_kw にはあるが、誤差計算/比較には不要なので使わない

# ===== パス設定 =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

CONFIG_PATH = os.path.join(PROJECT_DIR, "config", "tsukuba_conf", "MMK_Mix_PV_24.py")

# 24h 学習の ckpt（あなたが出してたやつ）
CKPT_PATH = os.path.join(
    PROJECT_DIR,
    "save", "MMK_Mix_PV_24", "MMK_Mix_PV_24",
    "seed_2025", "version_1", "checkpoints",
    "epoch=epoch=28-step=step=10469.ckpt"
)

OUT_DIR = os.path.join(PROJECT_DIR, "results_24h_plot")
os.makedirs(OUT_DIR, exist_ok=True)


def load_exp_conf(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("config_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod.exp_conf


def main():
    exp_conf = load_exp_conf(CONFIG_PATH)

    # ===== DataModule（NPY）を test でロード =====
    from easytsf.runner.data_runner import NPYDataInterface
    data_module = NPYDataInterface(**exp_conf)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    # ===== モデルロード =====
    from traingraph import PeakRunner, get_model
    base_model = get_model(exp_conf["model_name"], exp_conf)
    runner = PeakRunner.load_from_checkpoint(
        CKPT_PATH, model=base_model, config=exp_conf, strict=False
    ).eval()

    device = torch.device("cpu")
    runner.to(device)

    # ===== test_loader を最後まで回して「最後のバッチ」を取得 =====
    last_batch = None
    for batch in test_loader:
        last_batch = batch
    if last_batch is None:
        raise RuntimeError("[ERROR] test_loader が空です。データ生成や config を確認してください。")

    var_x, marker_x, var_y, marker_y = [_.float() for _ in last_batch]

    # ===== バッチ内の最後サンプルを取る =====
    # shape: var_x (B, 24, 8), var_y (B, 24, 1) の想定
    var_x = var_x[-1:].to(device)
    marker_x = marker_x[-1:].to(device)
    var_y = var_y[-1:].to(device)

    # ===== 学習時と一致させる：GHI boost =====
    # compare_models_kw.py と同じ：var_x[:,:,2] を 1.3倍
    var_x_input = var_x.clone()
    var_x_input[:, :, 2] *= 1.3

    # ===== 推論 =====
    with torch.no_grad():
        outputs = runner(var_x_input, marker_x)
        pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs  # (1, 24, C)

    # ===== チャンネル取り出し（compare_models_kw と一致）=====
    # 真値: var_y[:,:,0]（PVターゲット）
    # 予測: pred[:,:,3]（PV出力をここに置いている前提）
    true_norm = var_y[0, :, 0].cpu().numpy()          # (24,)
    pred_norm = pred[0, :, 3].cpu().numpy()           # (24,)

    # ===== kWに変換 =====
    true_kw = true_norm * KW_FACTOR
    pred_kw = pred_norm * KW_FACTOR

    # ===== 物理的に負の発電はあり得ないので、表示用に 0 未満は 0 にクリップ（任意）=====
    true_kw_plot = np.clip(true_kw, 0.0, None)
    pred_kw_plot = np.clip(pred_kw, 0.0, None)

    # ===== CSV保存 =====
    csv_path = os.path.join(OUT_DIR, "mmk_24h_last_window_kw.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step(1-24)", "true_kW_raw", "pred_kW_raw", "true_kW_clipped", "pred_kW_clipped"])
        for i in range(24):
            w.writerow([i + 1, float(true_kw[i]), float(pred_kw[i]), float(true_kw_plot[i]), float(pred_kw_plot[i])])

    # ===== プロット =====
    steps = np.arange(1, 25)
    plt.figure(figsize=(10, 4))
    plt.plot(steps, true_kw_plot, label="True (kW)")
    plt.plot(steps, pred_kw_plot, label="Pred (kW)")
    plt.xlabel("Forecast step (t+1 ... t+24) [-]")
    plt.ylabel("PV power [kW]")
    plt.title("MMK_Mix 24h - Last window forecast (kW)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, "mmk_24h_last_window_kw.png")
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print("[SUCCESS] Saved last-window 24h kW plot")
    print(f"  PNG: {out_png}")
    print(f"  CSV: {csv_path}")
    print("[INFO] sanity check (peak kW)")
    print(f"  True peak ≈ {true_kw_plot.max():.3f} kW, Pred peak ≈ {pred_kw_plot.max():.3f} kW")


if __name__ == "__main__":
    main()
