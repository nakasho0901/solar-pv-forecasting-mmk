# -*- coding: utf-8 -*-
"""
96in/96out の .npy から 24in/24out を生成する（多変量入力・単変量出力対応版）
"""

import os
import argparse
import numpy as np


def _load_npy(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] ファイルが見つかりません: {path}")
    arr = np.load(path)
    if arr.ndim != 3:
        raise ValueError(f"[ERROR] 3次元配列を想定: {path}, shape={arr.shape}")
    return arr


def _slice_xy(x96: np.ndarray, y96: np.ndarray, hist_len: int, pred_len: int):
    """
    x96: (S, 96, Nx)
    y96: (S, 96, Ny)  ※ Nx と Ny は一致しなくてよい
    """
    Sx, Lx, Nx = x96.shape
    Sy, Ly, Ny = y96.shape

    if Sx != Sy:
        raise ValueError(f"[ERROR] サンプル数が一致しません: x={x96.shape}, y={y96.shape}")
    if Lx < hist_len:
        raise ValueError(f"[ERROR] x の時間長が不足: Lx={Lx}, hist_len={hist_len}")
    if Ly < pred_len:
        raise ValueError(f"[ERROR] y の時間長が不足: Ly={Ly}, pred_len={pred_len}")

    x_new = x96[:, -hist_len:, :].copy()   # (S, 24, Nx)
    y_new = y96[:, :pred_len, :].copy()    # (S, 24, Ny)
    return x_new, y_new


def _slice_marker(marker96: np.ndarray, pred_len: int) -> np.ndarray:
    if marker96.ndim != 3:
        raise ValueError(f"[ERROR] marker は3次元を想定: shape={marker96.shape}")
    if marker96.shape[1] < pred_len:
        raise ValueError(f"[ERROR] marker の時間長が不足: shape={marker96.shape}")
    return marker96[:, :pred_len, :].copy()


def process_split(in_dir: str, out_dir: str, split: str, hist_len: int, pred_len: int):
    x96 = _load_npy(os.path.join(in_dir, f"{split}_x.npy"))
    y96 = _load_npy(os.path.join(in_dir, f"{split}_y.npy"))

    x_new, y_new = _slice_xy(x96, y96, hist_len, pred_len)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{split}_x.npy"), x_new)
    np.save(os.path.join(out_dir, f"{split}_y.npy"), y_new)

    print(f"[OK] {split}: x {x96.shape} -> {x_new.shape}, y {y96.shape} -> {y_new.shape}")

    marker_path = os.path.join(in_dir, f"{split}_marker_y.npy")
    if os.path.exists(marker_path):
        marker96 = _load_npy(marker_path)
        marker_new = _slice_marker(marker96, pred_len)
        np.save(os.path.join(out_dir, f"{split}_marker_y.npy"), marker_new)
        print(f"[OK] {split}: marker_y {marker96.shape} -> {marker_new.shape}")
    else:
        print(f"[SKIP] {split}: marker_y なし")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hist_len", type=int, default=24)
    ap.add_argument("--pred_len", type=int, default=24)
    args = ap.parse_args()

    for split in ["train", "val", "test"]:
        for suf in ["x", "y"]:
            p = os.path.join(args.in_dir, f"{split}_{suf}.npy")
            if not os.path.exists(p):
                raise FileNotFoundError(f"[ERROR] 必須ファイルがありません: {p}")

    print("[INFO] 96 → 24 時間データ生成")
    process_split(args.in_dir, args.out_dir, "train", args.hist_len, args.pred_len)
    process_split(args.in_dir, args.out_dir, "val", args.hist_len, args.pred_len)
    process_split(args.in_dir, args.out_dir, "test", args.hist_len, args.pred_len)

    print("[SUCCESS] 24in/24out データセット生成完了")


if __name__ == "__main__":
    main()
