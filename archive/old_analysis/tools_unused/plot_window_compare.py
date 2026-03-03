# -*- coding: utf-8 -*-
# 同じ idx の True に対して、2つの予測（before/after）を重ね描き
import argparse, os, numpy as np, matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--y", required=True)           # ローダ由来のY
ap.add_argument("--predA", required=True)       # before
ap.add_argument("--predB", required=True)       # after
ap.add_argument("--idx", type=int, required=True)
ap.add_argument("--out", default="results/compare_window.png")
args = ap.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
Y  = np.load(args.y)
A  = np.load(args.predA)
B  = np.load(args.predB)
n  = min(len(Y), len(A), len(B))
i  = min(args.idx, n-1)

for arr_name, arr in [("Y",Y),("A",A),("B",B)]:
    if arr.ndim == 3 and arr.shape[-1] == 1:
        locals()[arr_name] = arr[...,0]

maeA = float(np.mean(np.abs(A[i]-Y[i])))
maeB = float(np.mean(np.abs(B[i]-Y[i])))

plt.figure(figsize=(10,5))
plt.plot(Y[i], label="True", color="black", linewidth=2)
plt.plot(A[i], label=f"Before (MAE={maeA:.2f})", linestyle="--")
plt.plot(B[i], label=f"After  (MAE={maeB:.2f})", linestyle=":")
plt.title(f"Window {i}  (common_n={n})")
plt.xlabel("Time step (0–95)"); plt.ylabel("PV Power")
plt.grid(True); plt.legend(); plt.tight_layout()
plt.savefig(args.out, dpi=150)
print("Saved:", args.out, "  idx:", i, f"  MAE_before={maeA:.2f}  MAE_after={maeB:.2f}")
