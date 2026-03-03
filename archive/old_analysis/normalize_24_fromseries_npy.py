# -*- coding: utf-8 -*-
"""
tools/normalize_24_fromseries_npy.py

目的:
- 24_fromseries の .npy データを「PVだけ」正規化した新しい data_dir を作る。
- y は train_y の最大値 y_max[PV] で割る（無次元化）
- x も過去PVが入っているチャンネル（デフォルト x[:,:,0]）を同じ係数で割る
  ※入出力スケール不一致を避けるため重要

作成物:
- out_dir に train/val/test の x,y,marker をコピー/正規化して保存
- out_dir/scale_pv.txt に y_max を保存（逆正規化や再現用）

使い方例:
python tools\\normalize_24_fromseries_npy.py --in_dir "dataset_PV\\processed_24_fromseries_hardzero_v2" --out_dir "dataset_PV\\processed_24_fromseries_hardzero_v2_norm" --pv_x_index 0
"""

import os
import argparse
import numpy as np


def _load(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] not found: {path}")
    return np.load(path)


def _save(path: str, arr: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--pv_x_index", type=int, default=0, help="x の中で PV が入っている特徴量index（通常0）")
    args = parser.parse_args()

    in_dir = args.in_dir
    out_dir = args.out_dir
    pv_x_index = int(args.pv_x_index)

    os.makedirs(out_dir, exist_ok=True)

    # 入力一覧
    files = [
        "train_x.npy", "train_y.npy", "train_marker_y.npy",
        "val_x.npy",   "val_y.npy",   "val_marker_y.npy",
        "test_x.npy",  "test_y.npy",  "test_marker_y.npy",
    ]

    # まず train_y からスケール決定
    train_y = _load(os.path.join(in_dir, "train_y.npy")).astype(np.float32)  # (N,24,1)
    y_max = float(np.max(train_y))
    if y_max <= 0.0:
        raise ValueError(f"[ERROR] y_max must be > 0, but got {y_max}")

    # scale を保存（PV単位[PV] → 正規化は無次元）
    with open(os.path.join(out_dir, "scale_pv.txt"), "w", encoding="utf-8") as f:
        f.write(f"y_max_pv = {y_max}\n")
    print(f"[INFO] y_max (scale) = {y_max} [PV]  -> y_norm = y / y_max (dimensionless)")

    # 各 split を処理
    for split in ["train", "val", "test"]:
        x_path = os.path.join(in_dir, f"{split}_x.npy")
        y_path = os.path.join(in_dir, f"{split}_y.npy")
        my_path = os.path.join(in_dir, f"{split}_marker_y.npy")

        x = _load(x_path).astype(np.float32)  # (N,24,8) 想定
        y = _load(y_path).astype(np.float32)  # (N,24,1)
        my = _load(my_path).astype(np.float32)  # (N,24,1) 0/1

        # shapeチェック（壊れてたら早めに落とす）
        if x.ndim != 3:
            raise ValueError(f"[ERROR] {split}_x ndim must be 3, got {x.ndim}, shape={x.shape}")
        if y.ndim != 3:
            raise ValueError(f"[ERROR] {split}_y ndim must be 3, got {y.ndim}, shape={y.shape}")
        if my.ndim != 3:
            raise ValueError(f"[ERROR] {split}_marker_y ndim must be 3, got {my.ndim}, shape={my.shape}")

        if pv_x_index < 0 or pv_x_index >= x.shape[2]:
            raise ValueError(f"[ERROR] pv_x_index={pv_x_index} out of range for x.shape={x.shape}")

        # 正規化（PVだけ）
        x_norm = x.copy()
        x_norm[:, :, pv_x_index] = x_norm[:, :, pv_x_index] / y_max
        y_norm = y / y_max

        # 保存
        _save(os.path.join(out_dir, f"{split}_x.npy"), x_norm)
        _save(os.path.join(out_dir, f"{split}_y.npy"), y_norm)
        _save(os.path.join(out_dir, f"{split}_marker_y.npy"), my)

        print(f"[INFO] saved {split}: x_norm, y_norm, marker_y -> {out_dir}")

    print("[INFO] DONE.")


if __name__ == "__main__":
    main()
