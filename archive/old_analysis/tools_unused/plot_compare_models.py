# -*- coding: utf-8 -*-
"""
MMK vs iTransformer vs True の予測比較グラフ
逆正規化対応（mean/std, min/max, sklearn min_/scale_ 全対応）
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


# --- scaler 自動判定 ---
def load_scaler(path):
    d = np.load(path)
    keys = list(d.files)
    print("Scaler keys:", keys)

    # --- sklearn MinMaxScaler ---
    if "min_" in keys and "scale_" in keys:
        return ("sklearn_minmax", d["min_"], d["scale_"])

    # --- 自作 min/max scaler ---
    if "min" in keys and "max" in keys:
        return ("minmax", d["min"], d["max"])

    # --- StandardScaler (mean/std) ---
    if "mean" in keys and "std" in keys:
        return ("standard", d["mean"], d["std"])

    raise ValueError(f"Unsupported scaler format: keys={keys}")


# --- inverse transform ---
def inverse_transform(data, scaler):
    t, a, b = scaler

    if t == "sklearn_minmax":
        # x = scaled / scale + min
        return data / b + a

    if t == "minmax":
        # x = scaled*(max-min) + min
        return data * (b - a) + a

    if t == "standard":
        # x = scaled*std + mean
        return data * b + a

    raise ValueError("Unknown scaler type:", t)


def main(true_path, mmk_path, itr_path, index, out_dir, title):
    # --- load arrays ---
    y_true_all = np.load(true_path)
    y_mmk_all = np.load(mmk_path)
    y_itr_all = np.load(itr_path)

    # --- scaler load ---
    scaler = load_scaler("dataset_PV/processed/scaler_y.npz")

    # --- pick sample ---
    y_true = y_true_all[index, :, 0]
    y_mmk  = y_mmk_all[index, :, 0]
    y_itr  = y_itr_all[index, :, 0]

    # --- inverse ---
    y_true_inv = inverse_transform(y_true, scaler)
    y_mmk_inv  = inverse_transform(y_mmk, scaler)
    y_itr_inv  = inverse_transform(y_itr, scaler)

    # --- plot ---
    plt.figure(figsize=(14, 5))
    plt.plot(y_true_inv, label="True",  color="blue", linewidth=2)
    plt.plot(y_mmk_inv,  label="MMK",   color="orange", linewidth=2)
    plt.plot(y_itr_inv,  label="iTransformer", color="green", linewidth=2)

    plt.title(title)
    plt.xlabel("Hour (+ahead)")
    plt.ylabel("PV (original unit)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"compare_sample_{index}.png")
    plt.savefig(out_file)
    plt.show()

    # save CSV
    csv = np.vstack([y_true_inv, y_mmk_inv, y_itr_inv]).T
    csv_file = os.path.join(out_dir, f"compare_sample_{index}.csv")
    np.savetxt(csv_file, csv, delimiter=",", header="true,mmk,itr", comments="")

    print("Saved image:", out_file)
    print("Saved csv:", csv_file)


# --- CLI ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--true", required=True)
    parser.add_argument("--mmk", required=True)
    parser.add_argument("--itr", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out", default="results_compare")
    parser.add_argument("--title", default="MMK vs iTr Comparison")
    args = parser.parse_args()

    main(args.true, args.mmk, args.itr, args.index, args.out, args.title)
