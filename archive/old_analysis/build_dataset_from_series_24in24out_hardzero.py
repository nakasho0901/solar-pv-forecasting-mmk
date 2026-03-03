# -*- coding: utf-8 -*-
"""
連続時系列CSV（例: pv_weather_hourly.csv）から、
24in/24out の窓切り済み .npy（train/val/test）を生成するスクリプト。

この修正版のポイント（重要）
- hardzero を「系列全体」ではなく「各窓の未来ステップごと」に適用する
  => 未来の GHI(ghi_future) に基づき、Y[t] を 0 に強制する
- marker_y も「未来ステップごと」に確実に 0/1 を作る
  => marker_y[t] = 1 (day) if ghi_future[t] > threshold else 0 (night)

これにより、
- 入力(過去)のGHIと教師(未来)のPVが時刻ズレして混ざることがなくなる
- 検証も marker_y を使えば直感的にできる（nightは marker_y==0）

実行例（CMDワンライナー）
python tools\\build_dataset_from_series_24in24out_hardzero.py ^
  --csv dataset_PV\\prepared\\pv_weather_hourly.csv ^
  --out_dir dataset_PV\\processed_24_fromseries_hardzero ^
  --hist_len 24 --pred_len 24 ^
  --val_ratio 0.1 --test_ratio 0.1 ^
  --ghi_col ghi_wm2 --y_col pv_kwh --ghi_threshold 1.0 --dropna
"""

import argparse
import os
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True, help="入力CSV（連続時系列）例: pv_weather_hourly.csv")
    p.add_argument("--out_dir", type=str, required=True, help="出力先ディレクトリ")
    p.add_argument("--hist_len", type=int, default=24)
    p.add_argument("--pred_len", type=int, default=24)

    p.add_argument("--time_col", type=str, default="time")
    p.add_argument("--y_col", type=str, default="pv_kwh")
    p.add_argument("--ghi_col", type=str, default="ghi_wm2")
    p.add_argument("--day_col", type=str, default="is_daylight")  # 互換のため残す（ただし marker_y は未来GHI優先）

    p.add_argument("--ghi_threshold", type=float, default=1.0, help="夜判定のしきい値（ghi<=thresholdなら夜）")

    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)

    # 8変数を明示したい場合（カンマ区切り）
    p.add_argument("--x_cols", type=str, default="", help="入力に使う列名をカンマ区切りで指定（指定しない場合は自動選択）")

    # 欠損の扱い
    p.add_argument("--dropna", action="store_true", help="欠損を含む行を削除（推奨）")

    return p.parse_args()


def choose_x_cols(df: pd.DataFrame, x_cols_arg: str, var_num: int = 8):
    """
    入力特徴量（X）として使う列を var_num 個選ぶ。
    指定があればそれを優先、無ければ priority から存在する列を順に拾う。
    """
    if x_cols_arg.strip():
        cols = [c.strip() for c in x_cols_arg.split(",") if c.strip()]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"[ERROR] 指定された x_cols がCSVに存在しません: {missing}\n"
                f"CSV columns: {list(df.columns)}"
            )
        if len(cols) != var_num:
            raise ValueError(f"[ERROR] x_cols は {var_num} 個にしてください。現在 {len(cols)} 個: {cols}")
        return cols

    # 自動選択（pv_weather_hourly.csv 想定の優先順位）
    priority = [
        "temp_c", "rh_pct", "ghi_wm2",
        "lag_pv_kwh_1h", "lag_pv_kwh_2h", "lag_pv_kwh_3h", "lag_pv_kwh_24h",
        "roll_pv_kwh_24h_mean",
        "lag_ghi_wm2_1h", "lag_ghi_wm2_24h",
        "roll_ghi_wm2_24h_mean",
    ]
    picked = [c for c in priority if c in df.columns]

    if len(picked) < var_num:
        # 足りない場合は数値列から補充（time/y/周期特徴などを除外）
        exclude = set([
            "time", "datetime",
            "pv_kwh", "pv_kWh",
            "is_daylight",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos"
        ])
        numeric_cols = [
            c for c in df.columns
            if (c not in exclude) and pd.api.types.is_numeric_dtype(df[c])
        ]
        numeric_cols = [c for c in numeric_cols if c not in picked]
        picked.extend(numeric_cols)

    if len(picked) < var_num:
        raise ValueError(
            f"[ERROR] 入力特徴量が {var_num} 個集められませんでした。\n"
            f"候補: {picked}\nCSV columns: {list(df.columns)}"
        )

    return picked[:var_num]


def build_windows_with_future_hardzero(
    x: np.ndarray,
    y: np.ndarray,
    ghi: np.ndarray,
    hist_len: int,
    pred_len: int,
    ghi_threshold: float
):
    """
    x:   (T, var_num)
    y:   (T, 1)
    ghi: (T,)  ※時刻整列済み

    return:
      X:  (N, hist_len, var_num)
      Y:  (N, pred_len, 1)  ※未来GHIで hardzero 済み
      MY: (N, pred_len, 1)  ※未来GHIから作った day/night マスク（day=1, night=0）
    """
    T = x.shape[0]
    N = T - (hist_len + pred_len) + 1
    if N <= 0:
        raise ValueError(f"[ERROR] データ長が短すぎます。T={T}, hist_len={hist_len}, pred_len={pred_len}")

    var_num = x.shape[1]
    X = np.zeros((N, hist_len, var_num), dtype=np.float32)
    Y = np.zeros((N, pred_len, 1), dtype=np.float32)
    MY = np.zeros((N, pred_len, 1), dtype=np.float32)

    for i in range(N):
        # 入力（過去）
        X[i] = x[i:i + hist_len]

        # 教師（未来）
        y_future = y[i + hist_len:i + hist_len + pred_len].copy()          # (pred_len, 1)
        ghi_future = ghi[i + hist_len:i + hist_len + pred_len].copy()      # (pred_len,)

        # 未来ステップごとに hardzero（ここが修正の核心）
        night = (ghi_future <= ghi_threshold)
        y_future[night, 0] = 0.0

        # marker_y も未来GHIで必ず整合したものを作る（day=1, night=0）
        marker_future = (~night).astype(np.float32).reshape(pred_len, 1)

        Y[i] = y_future
        MY[i] = marker_future

    return X, Y, MY


def split_by_time(X, Y, MY, val_ratio: float, test_ratio: float):
    """
    時系列順のまま train/val/test に分割（shuffleしない）
    """
    N = X.shape[0]
    n_test = int(N * test_ratio)
    n_val = int(N * val_ratio)
    n_train = N - n_val - n_test
    if n_train <= 0:
        raise ValueError(f"[ERROR] split比率が不適切です。N={N}, train={n_train}, val={n_val}, test={n_test}")

    X_tr, Y_tr, MY_tr = X[:n_train], Y[:n_train], MY[:n_train]
    X_va, Y_va, MY_va = X[n_train:n_train + n_val], Y[n_train:n_train + n_val], MY[n_train:n_train + n_val]
    X_te, Y_te, MY_te = X[n_train + n_val:], Y[n_train + n_val:], MY[n_train + n_val:]

    return (X_tr, Y_tr, MY_tr), (X_va, Y_va, MY_va), (X_te, Y_te, MY_te)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)

    if args.time_col not in df.columns:
        raise ValueError(f"[ERROR] time_col='{args.time_col}' がCSVにありません。columns={list(df.columns)}")

    # timeでソート（念のため）
    df[args.time_col] = pd.to_datetime(df[args.time_col])
    df = df.sort_values(args.time_col).reset_index(drop=True)

    if args.dropna:
        before = len(df)
        df = df.dropna().reset_index(drop=True)
        print(f"[INFO] dropna: {before} -> {len(df)} rows")

    if args.y_col not in df.columns:
        raise ValueError(f"[ERROR] y_col='{args.y_col}' がCSVにありません。columns={list(df.columns)}")
    if args.ghi_col not in df.columns:
        raise ValueError(f"[ERROR] ghi_col='{args.ghi_col}' がCSVにありません。columns={list(df.columns)}")

    x_cols = choose_x_cols(df, args.x_cols, var_num=8)
    print("[INFO] Selected x_cols (var_num=8):")
    for c in x_cols:
        print("  -", c)

    # X, y, ghi を作る（同じ時刻軸で整列済みの前提）
    x = df[x_cols].astype(float).to_numpy().astype(np.float32)             # (T, 8)
    y = df[[args.y_col]].astype(float).to_numpy().astype(np.float32)       # (T, 1)
    ghi = df[args.ghi_col].astype(float).to_numpy().astype(np.float32)     # (T,)

    # 窓切り（未来GHIに基づく hardzero をここで適用）
    X, Y, MY = build_windows_with_future_hardzero(
        x=x,
        y=y,
        ghi=ghi,
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        ghi_threshold=args.ghi_threshold
    )

    # 分割（時間順）
    (X_tr, Y_tr, MY_tr), (X_va, Y_va, MY_va), (X_te, Y_te, MY_te) = split_by_time(
        X, Y, MY, args.val_ratio, args.test_ratio
    )

    # 保存
    np.save(os.path.join(args.out_dir, "train_x.npy"), X_tr)
    np.save(os.path.join(args.out_dir, "train_y.npy"), Y_tr)
    np.save(os.path.join(args.out_dir, "train_marker_y.npy"), MY_tr)

    np.save(os.path.join(args.out_dir, "val_x.npy"), X_va)
    np.save(os.path.join(args.out_dir, "val_y.npy"), Y_va)
    np.save(os.path.join(args.out_dir, "val_marker_y.npy"), MY_va)

    np.save(os.path.join(args.out_dir, "test_x.npy"), X_te)
    np.save(os.path.join(args.out_dir, "test_y.npy"), Y_te)
    np.save(os.path.join(args.out_dir, "test_marker_y.npy"), MY_te)

    print("[SUCCESS] 24in/24out dataset built from series (future-step hardzero).")
    print(f"  out_dir : {args.out_dir}")
    print(f"  train_x : {X_tr.shape}, train_y : {Y_tr.shape}, train_marker_y : {MY_tr.shape}")
    print(f"  val_x   : {X_va.shape}, val_y   : {Y_va.shape}, val_marker_y   : {MY_va.shape}")
    print(f"  test_x  : {X_te.shape}, test_y  : {Y_te.shape}, test_marker_y  : {MY_te.shape}")


if __name__ == "__main__":
    main()
