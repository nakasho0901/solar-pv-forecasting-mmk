# -*- coding: utf-8 -*-
"""
hardzero 結果の比較プロット（完全版）
- dataset_PV/processed_hardzero の scaler_y.npz で kW に逆変換
- marker_y (GHI未来) > threshold を「昼」として、昼のみ誤差も算出
- iTransformer vs MMK の予測を以下で可視化：
  (1) ある1サンプル(index)の 96点時系列：True / iTr / MMK（昼点を強調）
  (2) 昼の誤差 |pred-true| の分布（ヒストグラム）
  (3) MAE/MSE（全体・昼のみ）の棒グラフ
出力：out_dir に png を保存
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def load_scaler(path: str):
    sc = np.load(path)
    mean = float(np.atleast_1d(sc["mean"])[0])
    std = float(np.atleast_1d(sc["std"])[0])
    return mean, std


def squeeze_y(a: np.ndarray) -> np.ndarray:
    """[N,T,1] or [N,T] -> [N,T]"""
    return a[:, :, 0] if a.ndim == 3 else a


def inv_z(z: np.ndarray, mean: float, std: float) -> np.ndarray:
    """標準化空間 -> kW"""
    return z * std + mean


def metrics(y: np.ndarray, p: np.ndarray, mask: np.ndarray):
    """mask=True のみで評価（mask無し全体は mask=全True で渡す）"""
    diff = p[mask] - y[mask]
    mae = float(np.mean(np.abs(diff))) if diff.size else float("nan")
    mse = float(np.mean(diff ** 2)) if diff.size else float("nan")
    return mae, mse


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    # --- load scaler ---
    mean, std = load_scaler(args.scaler_y)

    # --- load arrays ---
    y = squeeze_y(np.load(args.true))
    p_itr = squeeze_y(np.load(args.itr))
    p_mmk = squeeze_y(np.load(args.mmk))
    marker = squeeze_y(np.load(args.marker_y))

    # --- inverse transform to kW ---
    yk = inv_z(y, mean, std)
    itrk = inv_z(p_itr, mean, std)
    mmkk = inv_z(p_mmk, mean, std)

    # --- day mask (same shape [N,T]) ---
    day = marker > args.threshold
    all_mask = np.ones_like(day, dtype=bool)

    # --- metrics ---
    itr_all_mae, itr_all_mse = metrics(yk, itrk, all_mask)
    mmk_all_mae, mmk_all_mse = metrics(yk, mmkk, all_mask)
    itr_day_mae, itr_day_mse = metrics(yk, itrk, day)
    mmk_day_mae, mmk_day_mse = metrics(yk, mmkk, day)

    day_ratio = float(np.mean(day))

    # ============ (1) Sample time-series ============
    idx = args.index
    if idx < 0 or idx >= yk.shape[0]:
        raise ValueError(f"--index が範囲外です: index={idx}, N={yk.shape[0]}")

    t = np.arange(yk.shape[1])
    day_t = day[idx]

    plt.figure()
    plt.plot(t, yk[idx], label="True [kW]")
    plt.plot(t, itrk[idx], label="iTransformer [kW]")
    plt.plot(t, mmkk[idx], label="MMK [kW]")

    # 昼の点だけを強調（True）
    plt.scatter(t[day_t], yk[idx][day_t], s=20, label="Day points (True)")

    plt.title(args.title + f" (index={idx})")
    plt.xlabel("Horizon step (0..95) [hour-step]")
    plt.ylabel("PV power [kW]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out1 = os.path.join(args.out_dir, f"timeseries_index{idx}.png")
    plt.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close()

    # ============ (2) Day absolute error histogram ============
    abs_err_itr_day = np.abs(itrk[day] - yk[day])
    abs_err_mmk_day = np.abs(mmkk[day] - yk[day])

    plt.figure()
    plt.hist(abs_err_itr_day, bins=args.bins, alpha=0.6, label="iTransformer |err| (Day)")
    plt.hist(abs_err_mmk_day, bins=args.bins, alpha=0.6, label="MMK |err| (Day)")
    plt.title(args.title + " - Day |error| distribution")
    plt.xlabel("|Prediction - True| [kW]")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out2 = os.path.join(args.out_dir, "day_abs_error_hist.png")
    plt.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close()

    # ============ (3) MAE/MSE bar chart ============
    labels = ["ALL_MAE", "ALL_MSE", "DAY_MAE", "DAY_MSE"]
    itr_vals = [itr_all_mae, itr_all_mse, itr_day_mae, itr_day_mse]
    mmk_vals = [mmk_all_mae, mmk_all_mse, mmk_day_mae, mmk_day_mse]

    x = np.arange(len(labels))
    w = 0.38

    plt.figure()
    plt.bar(x - w/2, itr_vals, width=w, label="iTransformer")
    plt.bar(x + w/2, mmk_vals, width=w, label="MMK")
    plt.xticks(x, labels)
    plt.title(args.title + f" - Metrics (day_ratio={day_ratio:.3f})")
    plt.ylabel("Value [kW] or [(kW)^2]")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    out3 = os.path.join(args.out_dir, "metrics_bar.png")
    plt.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close()

    # --- print summary ---
    print("=== Saved plots ===")
    print(out1)
    print(out2)
    print(out3)
    print("=== Metrics (kW) ===")
    print(f"day_ratio={day_ratio:.6f} (threshold={args.threshold})")
    print(f"iTr  ALL: MAE={itr_all_mae:.6f} [kW], MSE={itr_all_mse:.6f} [(kW)^2]")
    print(f"iTr  DAY: MAE={itr_day_mae:.6f} [kW], MSE={itr_day_mse:.6f} [(kW)^2]")
    print(f"MMK  ALL: MAE={mmk_all_mae:.6f} [kW], MSE={mmk_all_mse:.6f} [(kW)^2]")
    print(f"MMK  DAY: MAE={mmk_day_mae:.6f} [kW], MSE={mmk_day_mse:.6f} [(kW)^2]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--true", required=True, help="true_test.npy or test_y.npy (normalized)")
    ap.add_argument("--itr", required=True, help="iTransformer test_preds.npy (normalized)")
    ap.add_argument("--mmk", required=True, help="MMK test_preds.npy (normalized)")
    ap.add_argument("--marker_y", required=True, help="test_marker_y.npy (GHI future)")
    ap.add_argument("--scaler_y", default=r"dataset_PV\processed_hardzero\scaler_y.npz", help="scaler_y.npz")
    ap.add_argument("--threshold", type=float, default=1.0, help="day threshold for marker_y (GHI)")
    ap.add_argument("--index", type=int, default=0, help="which sample index to plot time-series")
    ap.add_argument("--bins", type=int, default=60, help="hist bins")
    ap.add_argument("--out_dir", default="results_hardzero_compare", help="output directory")
    ap.add_argument("--title", default="hardzero: iTransformer vs MMK", help="plot title")
    args = ap.parse_args()
    main(args)
