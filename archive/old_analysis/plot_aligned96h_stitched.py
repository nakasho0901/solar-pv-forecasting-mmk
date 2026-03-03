# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="aligned_means_night0.npz path")
    ap.add_argument("--outdir", default="results_aligned96h_stitched")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    z = np.load(args.npz)
    mean_true = z["mean_true"]  # (4,24)
    mean_mmk  = z["mean_mmk"]   # (4,24)
    mean_itr  = z["mean_itr"]   # (4,24)

    # 96hに連結
    y_true = mean_true.reshape(-1)  # (96,)
    y_mmk  = mean_mmk.reshape(-1)
    y_itr  = mean_itr.reshape(-1)

    x = np.arange(96)

    plt.figure(figsize=(12, 4))
    plt.plot(x, y_true, label="True mean (stitched 96h)", linewidth=2)
    plt.plot(x, y_mmk,  label="MMK pred mean (stitched 96h)", linewidth=2)
    plt.plot(x, y_itr,  label="iTr pred mean (stitched 96h)", linewidth=2)
    plt.xlabel("Horizon step [hour] (stitched 0..95)")
    plt.ylabel("PV")
    plt.title("Aligned 96h Mean (Night0, JST) - stitched 96h")
    plt.grid(True)
    plt.legend()
    outpng = os.path.join(args.outdir, "aligned96h_stitched_night0.png")
    plt.savefig(outpng, dpi=150, bbox_inches="tight")
    plt.close()

    print("[OK] saved:", os.path.abspath(outpng))

if __name__ == "__main__":
    main()
