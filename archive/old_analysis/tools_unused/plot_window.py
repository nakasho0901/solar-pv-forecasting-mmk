# -*- coding: utf-8 -*-
# 特定ウィンドウ idx の True vs Pred を保存（Yはローダ由来を推奨）
import argparse, os
import numpy as np
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--y",   required=True,  help="path to test_y (e.g., .../test_y_from_loader.npy)")
ap.add_argument("--pred",required=True,  help="path to test_preds.npy")
ap.add_argument("--idx", type=int,      required=True,  help="window index to plot")
ap.add_argument("--out", default="results/plot_window.png")
args = ap.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
Y = np.load(args.y); P = np.load(args.pred)
n = min(len(Y), len(P))
idx = min(args.idx, n-1)

# (N,96,1) → (N,96)
if Y.ndim == 3 and Y.shape[-1] == 1: Y = Y[...,0]
if P.ndim == 3 and P.shape[-1] == 1: P = P[...,0]

mae = float(np.mean(np.abs(P[idx]-Y[idx])))
plt.figure(figsize=(10,5))
plt.plot(Y[idx], label="True", color="black", linewidth=2)
plt.plot(P[idx], label=f"Pred (MAE={mae:.2f})", linestyle="--")
plt.title(f"Window {idx}  (common_n={n})")
plt.xlabel("Time step (0–95)"); plt.ylabel("PV Power")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(args.out, dpi=150)
print("Saved:", args.out, "  idx:", idx, "  MAE:", mae)
