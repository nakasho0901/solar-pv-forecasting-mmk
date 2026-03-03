#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd

def make_windows(data, lookback, horizon):
    X, Y = [], []
    for i in range(len(data) - lookback - horizon + 1):
        x = data[i:i+lookback]
        y = data[i+lookback:i+lookback+horizon, 0]  # 0列目=目的変数
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--lookback", type=int, default=96)
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.data)
    if "pv_kwh" not in df.columns:
        raise ValueError("CSVに 'pv_kwh' 列がありません。")

    values = df.drop(columns=["time"]).values.astype(np.float32)
    n = len(values)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val

    train = values[:n_train]
    val = values[n_train - args.lookback:n_train + n_val]
    test = values[n_train + n_val - args.lookback:]

    os.makedirs(args.out, exist_ok=True)

    for name, split in zip(["train", "val", "test"], [train, val, test]):
        X, Y = make_windows(split, args.lookback, args.horizon)
        np.save(os.path.join(args.out, f"{name}_x.npy"), X)
        np.save(os.path.join(args.out, f"{name}_y.npy"), Y)
        print(f"[{name}] X: {X.shape}, Y: {Y.shape}")

    print(f"[OK] Saved splits to {args.out}")

if __name__ == "__main__":
    main()
