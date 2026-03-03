# tools/build_dataset_96in96out.py
# -*- coding: utf-8 -*-
"""
気象 + 過去PV を入力（96h）として、未来96hのPV発電量を予測するための
学習データセットを構築するスクリプト（リーク完全防止）。
- 列名を柔軟に指定可能（--pv-time-col, --pv-col など）
- タイムゾーンが混在していても、両方を「naive JST」に統一してマージ
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd


# ===============================================================
# 補助関数群
# ===============================================================
@dataclass
class SplitIdx:
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


def ensure_dir(path: str):
    """出力ディレクトリが無ければ作成"""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def to_naive_jst(series: pd.Series) -> pd.Series:
    """
    時刻列を「タイムゾーン付き/なしに関わらず」読み取り、
    ・tz-awareなら Asia/Tokyo に変換して tz情報を外す（naive JST）
    ・tzなしなら そのまま（JSTで記録されている前提）
    に統一する。
    """
    s = pd.to_datetime(series, errors="coerce", utc=False)
    # tz情報の有無で分岐（tz-awareなら tz_convert → tz_localize(None)）
    try:
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
    except TypeError:
        # すでに tzなし（naive）なら何もしない
        pass
    return s


def add_time_features(df: pd.DataFrame, time_col: str = "time") -> pd.DataFrame:
    """sin/cos の時間特徴を追加"""
    dt = pd.to_datetime(df[time_col])
    hour = dt.dt.hour.values.astype(float)   # [h]
    doy  = dt.dt.dayofyear.values.astype(float)  # [1..366]
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    df["sin_doy"]  = np.sin(2 * np.pi * doy  / 365.25)
    df["cos_doy"]  = np.cos(2 * np.pi * doy  / 365.25)
    return df


def night_zero_correction(df, ghi_col, pv_col, threshold):
    """ghiが閾値以下のときpvを0に補正（物理整合性）"""
    df.loc[df[ghi_col] <= threshold, pv_col] = 0.0
    return df


def make_windows(df, feature_cols, target_col, hist_len, fut_len):
    """過去hist_len → 未来fut_lenのスライディング窓を作成"""
    X, Y, idx = [], [], []
    vals_feat = df[feature_cols].to_numpy(np.float32)
    vals_tgt  = df[target_col].to_numpy(np.float32)
    total = len(df)
    max_start = total - hist_len - fut_len
    for i in range(max_start + 1):
        X.append(vals_feat[i:i+hist_len])
        Y.append(vals_tgt[i+hist_len:i+hist_len+fut_len])
        idx.append(i+hist_len-1)
    return np.stack(X), np.stack(Y), np.array(idx)


def compute_splits(n, ratio=(0.8, 0.1, 0.1)) -> SplitIdx:
    """8:1:1で分割（時系列順・シャッフルなし）"""
    n_train = int(n * ratio[0])
    n_val   = int(n * ratio[1])
    n_test  = n - n_train - n_val
    return SplitIdx(0, n_train, n_train, n_train+n_val, n_train+n_val, n)


def fit_standardizer(arr):
    """平均・標準偏差を算出（ゼロ割防止）"""
    if arr.ndim == 3:
        mean, std = arr.mean((0,1)), arr.std((0,1))
        std[std == 0] = 1
    else:
        mean, std = arr.mean(), arr.std()
        if std == 0:
            std = 1
    return mean, std


def standardize(arr, mean, std):
    """標準化"""
    return (arr - mean) / std


# ===============================================================
# メイン処理
# ===============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("-w", "--weather_csv", required=True, help="weather_processed.csv へのパス")
    p.add_argument("-p", "--pv_csv",      required=True, help="pv_hourly_energy_total.csv へのパス")
    p.add_argument("-o", "--out_dir",     required=True, help="出力ディレクトリ")
    # 列名指定（あなたの実データに合わせて上書き可）
    p.add_argument("--wx-time-col", default="time")
    p.add_argument("--pv-time-col", default="time")
    p.add_argument("--pv-col",      default="pv")
    p.add_argument("--wx-temp-col", default="temp_c")
    p.add_argument("--wx-rh-col",   default="rh_pct")
    p.add_argument("--wx-ghi-col",  default="ghi_wm2")
    # パラメータ
    p.add_argument("--ghi-threshold", type=float, default=1.0)
    p.add_argument("--hist-len", type=int, default=96)
    p.add_argument("--fut-len",  type=int, default=96)
    args = p.parse_args()

    ensure_dir(args.out_dir)

    # ---------- CSV読み込み ----------
    weather = pd.read_csv(args.weather_csv)
    pv      = pd.read_csv(args.pv_csv)

    # ---------- 列名を統一（内部名へrename） ----------
    weather = weather.rename(columns={
        args.wx_time_col: "time",
        args.wx_temp_col: "temp_c",
        args.wx_rh_col:   "rh_pct",
        args.wx_ghi_col:  "ghi_wm2"
    })
    pv = pv.rename(columns={
        args.pv_time_col: "time",
        args.pv_col:      "pv"
    })

    # ---------- タイムゾーン混在を「naive JST」に統一 ----------
    weather["time"] = to_naive_jst(weather["time"])
    pv["time"]      = to_naive_jst(pv["time"])

    # 念のためソート（asofマージは昇順が必須）
    weather = weather.sort_values("time").reset_index(drop=True)
    pv      = pv.sort_values("time").reset_index(drop=True)

    # ---------- timeで厳密マージ（完全一致前提、ズレがあるなら toleranceを広げる） ----------
    df = pd.merge_asof(
        weather, pv,
        on="time", direction="nearest", tolerance=pd.Timedelta("0min")
    )

    # PVが欠損している行は落とす（不一致があった場合の保守的処理）
    df = df.dropna(subset=["pv"]).reset_index(drop=True)

    # ---------- 夜間ゼロ補正 + 時間特徴 ----------
    df = night_zero_correction(df, "ghi_wm2", "pv", args.ghi_threshold)
    df = add_time_features(df, time_col="time")

    # ---------- 特徴セット ----------
    features = ["temp_c", "rh_pct", "ghi_wm2", "pv", "sin_hour", "cos_hour", "sin_doy", "cos_doy"]

    # ---------- 96→96窓生成 ----------
    X_all, Y_all, idx = make_windows(df, features, "pv", args.hist_len, args.fut_len)
    if len(X_all) < 10:
        raise RuntimeError(f"サンプル数が少なすぎます（{len(X_all)}）。期間やhist/futを見直してください。")

    # ---------- 分割（8:1:1） ----------
    sp = compute_splits(len(X_all))
    Xtr, Ytr = X_all[sp.train_start:sp.train_end], Y_all[sp.train_start:sp.train_end]
    Xv,  Yv  = X_all[sp.val_start:sp.val_end],     Y_all[sp.val_start:sp.val_end]
    Xt,  Yt  = X_all[sp.test_start:sp.test_end],   Y_all[sp.test_start:sp.test_end]

    # ---------- 標準化（fitは学習のみ） ----------
    mx, sx = fit_standardizer(Xtr)
    my, sy = fit_standardizer(Ytr)
    Xtr, Xv, Xt = standardize(Xtr, mx, sx), standardize(Xv, mx, sx), standardize(Xt, mx, sx)
    Ytr, Yv, Yt = standardize(Ytr, my, sy), standardize(Yv, my, sy), standardize(Yt, my, sy)

    # ---------- 保存 ----------
    np.save(os.path.join(args.out_dir, "train_x.npy"), Xtr)
    np.save(os.path.join(args.out_dir, "train_y.npy"), Ytr)
    np.save(os.path.join(args.out_dir, "val_x.npy"),   Xv)
    np.save(os.path.join(args.out_dir, "val_y.npy"),   Yv)
    np.save(os.path.join(args.out_dir, "test_x.npy"),  Xt)
    np.save(os.path.join(args.out_dir, "test_y.npy"),  Yt)
    np.savez(os.path.join(args.out_dir, "scaler_x.npz"), mean=mx, std=sx)
    np.savez(os.path.join(args.out_dir, "scaler_y.npz"), mean=my, std=sy)

    meta = {
        "features": features,
        "hist_len_h": args.hist_len,
        "fut_len_h":  args.fut_len,
        "counts": {"train": len(Xtr), "val": len(Xv), "test": len(Xt)},
        "ghi_threshold_Wm2": args.ghi_threshold,
        "note": "times normalized to naive JST before merge_asof"
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("=== Done ===")
    print(f"Saved to: {args.out_dir}")
    print(f"Shapes: X={X_all.shape}, Y={Y_all.shape}")


if __name__ == "__main__":
    main()
