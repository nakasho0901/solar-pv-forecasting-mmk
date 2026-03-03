# -*- coding: utf-8 -*-
"""
tools/plot_last4days_concat.py

目的:
- make_error_plots_24.py が保存した pred.npy / true.npy / day_mask.npy を読み込み
- 「最後の4日分」を 24ステップ×4 = 96ステップ に連結
- 実測(true)と予測(pred)を同じグラフに重ねて保存

前提:
- pred.npy, true.npy: shape = (N, 24, 1)
- day_mask.npy はあれば読み込む（無くてもOK）
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, required=True, help="pred.npy/true.npy があるフォルダ")
    parser.add_argument("--out", type=str, required=True, help="出力pngパス")
    parser.add_argument("--k_days", type=int, default=4, help="連結する日数（デフォルト4）")
    args = parser.parse_args()

    pred_path = os.path.join(args.indir, "pred.npy")
    true_path = os.path.join(args.indir, "true.npy")
    day_path  = os.path.join(args.indir, "day_mask.npy")

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        raise FileNotFoundError(f"[ERROR] pred.npy or true.npy not found in: {args.indir}")

    pred = np.load(pred_path)  # (N,24,1)
    true = np.load(true_path)  # (N,24,1)

    if pred.ndim != 3 or pred.shape[1] != 24 or pred.shape[2] != 1:
        raise ValueError(f"[ERROR] pred shape unexpected: {pred.shape} (expected (N,24,1))")
    if true.shape != pred.shape:
        raise ValueError(f"[ERROR] true shape mismatch: true={true.shape}, pred={pred.shape}")

    N = pred.shape[0]
    k = int(args.k_days)
    if N < k:
        raise ValueError(f"[ERROR] N={N} samples, but k_days={k} is too large.")

    # ===== 最後のk日（kサンプル）を取り出して連結 =====
    pred_last = pred[-k:, :, 0]  # (k,24)
    true_last = true[-k:, :, 0]  # (k,24)

    pred_concat = pred_last.reshape(-1)  # (k*24,)
    true_concat = true_last.reshape(-1)  # (k*24,)

    # day_mask があれば読み込んで、昼だけ薄く目印を付ける（任意）
    day_concat = None
    if os.path.exists(day_path):
        day = np.load(day_path)  # (N,24,1)
        if day.shape == pred.shape:
            day_last = day[-k:, :, 0]  # (k,24)
            day_concat = day_last.reshape(-1)  # (k*24,)

    # ===== プロット =====
    T = k * 24
    x = np.arange(T)

    plt.figure()
    plt.plot(x, true_concat, label="true")
    plt.plot(x, pred_concat, label="pred")

    # 昼(1)の領域をうっすら帯で表示（ある場合のみ）
    if day_concat is not None:
        # day==1 の区間を塗る（連続区間を探す）
        in_run = False
        start = 0
        for i in range(T):
            is_day = (day_concat[i] > 0.5)
            if is_day and not in_run:
                in_run = True
                start = i
            if (not is_day or i == T - 1) and in_run:
                end = i if not is_day else i + 1
                plt.axvspan(start, end, alpha=0.08)  # 色はデフォルト（指定しない）
                in_run = False

    plt.title(f"Last {k} days concat (24h forecast x {k})")
    plt.xlabel("time step (hour index)")
    plt.ylabel("PV (same unit as y)")
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(args.out)
    plt.close()

    print(f"[INFO] saved: {args.out}")
    print(f"[INFO] concat length = {T} points (={k}*24)")


if __name__ == "__main__":
    main()
