# -*- coding: utf-8 -*-
r"""
compare_mean96h_mmk_vs_itr.py  (Aligned 96h Mean / Night0)

目的:
- MMK / iTr / True を同時に比較
- 96h (4日) を Day1..Day4 に分割
- 各Dayについて「JST hour-of-day に整列」して平均（Aligned mean）
- Night0あり（marker_y があればそれを使用。無ければ true<=pv_eps でfallback）

入力（各 outdir について）:
優先的に以下のファイルを探します（どれかの組が揃えばOK）:
A) compare_models_meanplots.py の --save_arrays 形式（推奨）
    - arrays/MMK_pred_night0.npy, arrays/MMK_true.npy, arrays/MMK_t0.npy, arrays/MMK_marker_y.npy (任意)
    - arrays/iTr_pred_night0.npy, arrays/iTr_true.npy, arrays/iTr_t0.npy, arrays/iTr_marker_y.npy (任意)

B) plot_pred_vs_true 系の outdir 形式（ある程度互換）
    - pred_pv_test.npy または pred_pv_night0.npy
    - true_pv_test.npy または true_pv.npy
    - t0: test_t0.npy / t0.npy / *_t0.npy 等（見つかれば）
    - marker_y: test_marker_y.npy / marker_y.npy / *_marker_y.npy 等（見つかれば）

出力:
- aligned_day{1..4}_night0.png
- aligned_4days_night0.png (4枚を縦にまとめた図)
- aligned_means_night0.npz (数値も保存)
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# utils: find files robustly
# -----------------------------
def _try_paths(root, rel_candidates):
    for rel in rel_candidates:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            return p
    return None


def _find_arrays_dir(outdir: str):
    cand = os.path.join(outdir, "arrays")
    return cand if os.path.isdir(cand) else None


def load_outdir_data(outdir: str, name: str, pred_len: int = 96):
    """
    name: "MMK" or "iTr"
    return dict with:
      pred: (N, 96)
      true: (N, 96)
      t0:   (N,) datetime64 or None
      marker_y: (N, 96) float/bool or None
    """
    arrays_dir = _find_arrays_dir(outdir)

    # --- 1) pred ---
    pred_path = None
    if arrays_dir is not None:
        pred_path = _try_paths(arrays_dir, [f"{name}_pred_night0.npy", f"{name}_pred_raw.npy"])
    if pred_path is None:
        pred_path = _try_paths(outdir, ["pred_pv_night0.npy", "pred_pv_test.npy", "pred_test.npy", "pred.npy"])
    if pred_path is None:
        raise FileNotFoundError(f"[ERROR] pred not found in {outdir} (name={name})")

    pred = np.load(pred_path)

    # --- 2) true ---
    true_path = None
    if arrays_dir is not None:
        true_path = _try_paths(arrays_dir, [f"{name}_true.npy"])
    if true_path is None:
        true_path = _try_paths(outdir, ["true_pv_test.npy", "true_pv.npy", "true_test.npy", "true.npy"])
    if true_path is None:
        raise FileNotFoundError(f"[ERROR] true not found in {outdir} (name={name})")

    true = np.load(true_path)

    # --- shape normalize ---
    # accept (N,96) only
    if pred.ndim != 2 or true.ndim != 2:
        raise ValueError(f"[ERROR] pred/true must be 2D. pred={pred.shape}, true={true.shape}")
    if pred.shape[1] != pred_len or true.shape[1] != pred_len:
        raise ValueError(f"[ERROR] pred_len mismatch. pred={pred.shape}, true={true.shape}, expected_len={pred_len}")
    if pred.shape[0] != true.shape[0]:
        raise ValueError(f"[ERROR] N mismatch. pred={pred.shape}, true={true.shape}")

    # --- 3) t0 ---
    t0_path = None
    if arrays_dir is not None:
        t0_path = _try_paths(arrays_dir, [f"{name}_t0.npy"])
    if t0_path is None:
        # try common names under outdir (if you copied them)
        t0_path = _try_paths(outdir, ["test_t0.npy", "t0.npy"])
    t0 = np.load(t0_path) if t0_path is not None else None

    # --- 4) marker_y ---
    marker_y_path = None
    if arrays_dir is not None:
        marker_y_path = _try_paths(arrays_dir, [f"{name}_marker_y.npy"])
    if marker_y_path is None:
        marker_y_path = _try_paths(outdir, ["test_marker_y.npy", "marker_y.npy"])
    marker_y = np.load(marker_y_path) if marker_y_path is not None else None

    # marker_y normalize to (N,96)
    if marker_y is not None:
        if marker_y.ndim == 3 and marker_y.shape[-1] == 1:
            marker_y = marker_y[:, :, 0]
        if marker_y.ndim != 2:
            raise ValueError(f"[ERROR] marker_y must be (N,96) or (N,96,1). got {marker_y.shape}")
        if marker_y.shape != (pred.shape[0], pred_len):
            raise ValueError(f"[ERROR] marker_y shape mismatch. marker_y={marker_y.shape}, expected={(pred.shape[0], pred_len)}")

    return {
        "pred": pred.astype(np.float32),
        "true": true.astype(np.float32),
        "t0": t0,
        "marker_y": marker_y.astype(np.float32) if marker_y is not None else None,
        "paths": {"pred": pred_path, "true": true_path, "t0": t0_path, "marker_y": marker_y_path},
    }


# -----------------------------
# aligned mean (JST)
# -----------------------------
def t0_hour_jst(t0: np.ndarray, tz_offset_hours: int = 9) -> np.ndarray:
    """
    t0: numpy datetime64 array (N,)
    return: (N,) hour in [0..23] at JST
    """
    # convert to hours from epoch
    h_utc = t0.astype("datetime64[h]").astype("int64")
    h_jst = (h_utc + int(tz_offset_hours)) % 24
    return h_jst.astype(np.int32)


def make_night0(seq: np.ndarray, marker_y: np.ndarray, true: np.ndarray, pv_eps: float) -> np.ndarray:
    """
    seq: (N,96)
    marker_y: (N,96) or None
    fallback: true<=pv_eps => mask=0
    """
    if marker_y is not None:
        return seq * marker_y
    # fallback (markerが無い場合): trueが小さいところを夜とみなす
    mask = (true > pv_eps).astype(np.float32)
    return seq * mask


def aligned_96h_mean_by_day_hour(
    seq: np.ndarray,
    t0: np.ndarray,
    pred_len: int = 96,
    tz_offset_hours: int = 9
) -> np.ndarray:
    """
    seq: (N,96)
    t0:  (N,) datetime64
    return: mean[d, h] shape (4,24)
      d=0..3 (Day1..Day4), h=0..23 (JST hour-of-day)
    """
    if t0 is None:
        raise ValueError("[ERROR] aligned mean requires t0.npy (window start time).")

    N = seq.shape[0]
    hod0 = t0_hour_jst(t0, tz_offset_hours=tz_offset_hours)  # (N,)

    sumv = np.zeros((4, 24), dtype=np.float64)
    cnt = np.zeros((4, 24), dtype=np.int64)

    for i in range(N):
        base = int(hod0[i])
        for k in range(pred_len):
            d = k // 24  # 0..3
            h = (base + k) % 24
            sumv[d, h] += float(seq[i, k])
            cnt[d, h] += 1

    mean = sumv / np.maximum(cnt, 1)
    return mean.astype(np.float32)


# -----------------------------
# plotting
# -----------------------------
def plot_day(mean_true_24, mean_mmk_24, mean_itr_24, outpath, title):
    h = np.arange(24)
    plt.figure(figsize=(11, 4))
    plt.plot(h, mean_true_24, label="True mean (JST)", linewidth=2)
    plt.plot(h, mean_mmk_24, label="MMK pred mean (JST)", linewidth=2)
    plt.plot(h, mean_itr_24, label="iTr pred mean (JST)", linewidth=2)
    plt.xlabel("Hour of Day [hour] (JST)")
    plt.ylabel("PV")
    plt.title(title)
    plt.xticks(h)
    plt.grid(True)
    plt.legend()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_4days_stack(mean_true_4x24, mean_mmk_4x24, mean_itr_4x24, outpath, title_prefix):
    # 4つを縦にまとめる（研究スライドで便利）
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    h = np.arange(24)

    for d in range(4):
        ax = axes[d]
        ax.plot(h, mean_true_4x24[d], label="True mean (JST)", linewidth=2)
        ax.plot(h, mean_mmk_4x24[d], label="MMK pred mean (JST)", linewidth=2)
        ax.plot(h, mean_itr_4x24[d], label="iTr pred mean (JST)", linewidth=2)
        ax.set_ylabel("PV")
        ax.grid(True)
        ax.set_title(f"{title_prefix} Day{d+1}")

    axes[-1].set_xlabel("Hour of Day [hour] (JST)")
    axes[-1].set_xticks(h)
    axes[0].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmk_dir", required=True, help="MMK outdir (can contain arrays/)")
    ap.add_argument("--itr_dir", required=True, help="iTr outdir (can contain arrays/)")
    ap.add_argument("--outdir", default="results_compare_aligned96h")
    ap.add_argument("--pred_len", type=int, default=96)
    ap.add_argument("--tz_offset_hours", type=int, default=9, help="JST=+9")
    ap.add_argument("--pv_eps", type=float, default=0.5, help="fallback night detector if marker_y missing")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    mmk = load_outdir_data(args.mmk_dir, name="MMK", pred_len=args.pred_len)
    itr = load_outdir_data(args.itr_dir, name="iTr", pred_len=args.pred_len)

    # Night0あり（marker_y優先、無ければ true<=pv_eps でfallback）
    mmk_pred_n0 = make_night0(mmk["pred"], mmk["marker_y"], mmk["true"], pv_eps=args.pv_eps)
    itr_pred_n0 = make_night0(itr["pred"], itr["marker_y"], itr["true"], pv_eps=args.pv_eps)

    # TrueもNight0（同じmaskで0化：markerが無い場合は true<=pv_eps で0になる）
    mmk_true_n0 = make_night0(mmk["true"], mmk["marker_y"], mmk["true"], pv_eps=args.pv_eps)
    itr_true_n0 = make_night0(itr["true"], itr["marker_y"], itr["true"], pv_eps=args.pv_eps)

    # aligned meanは t0 必須
    if mmk["t0"] is None:
        raise FileNotFoundError("[ERROR] MMK side t0 not found. Need arrays/MMK_t0.npy (or test_t0.npy copied into outdir).")
    if itr["t0"] is None:
        raise FileNotFoundError("[ERROR] iTr side t0 not found. Need arrays/iTr_t0.npy (or test_t0.npy copied into outdir).")

    mean_true = aligned_96h_mean_by_day_hour(mmk_true_n0, mmk["t0"], args.pred_len, args.tz_offset_hours)
    mean_mmk  = aligned_96h_mean_by_day_hour(mmk_pred_n0, mmk["t0"], args.pred_len, args.tz_offset_hours)
    mean_itr  = aligned_96h_mean_by_day_hour(itr_pred_n0, itr["t0"], args.pred_len, args.tz_offset_hours)

    # day-wise plots
    for d in range(4):
        outpng = os.path.join(args.outdir, f"aligned_day{d+1}_night0.png")
        plot_day(
            mean_true[d], mean_mmk[d], mean_itr[d],
            outpath=outpng,
            title=f"Aligned 96h Mean (Night0, JST) Day{d+1}: pred mean vs true mean",
        )

    # stacked plot
    outpng_all = os.path.join(args.outdir, "aligned_4days_night0.png")
    plot_4days_stack(mean_true, mean_mmk, mean_itr, outpath=outpng_all, title_prefix="Aligned 96h Mean (Night0, JST)")

    # save numbers
    np.savez(
        os.path.join(args.outdir, "aligned_means_night0.npz"),
        mean_true=mean_true,
        mean_mmk=mean_mmk,
        mean_itr=mean_itr,
    )

    # simple log
    print("[OK] saved to:", os.path.abspath(args.outdir))
    print("[INFO] used files:")
    print("  MMK pred:", mmk["paths"]["pred"])
    print("  MMK true:", mmk["paths"]["true"])
    print("  MMK t0  :", mmk["paths"]["t0"])
    print("  MMK my  :", mmk["paths"]["marker_y"])
    print("  iTr pred:", itr["paths"]["pred"])
    print("  iTr true:", itr["paths"]["true"])
    print("  iTr t0  :", itr["paths"]["t0"])
    print("  iTr my  :", itr["paths"]["marker_y"])


if __name__ == "__main__":
    main()
