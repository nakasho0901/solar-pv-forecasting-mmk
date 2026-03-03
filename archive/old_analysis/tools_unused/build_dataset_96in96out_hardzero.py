# tools/build_dataset_96in96out_hardzero.py
# -*- coding: utf-8 -*-
"""
過去96h→未来96h の学習データを作成。
X: [temp_c, rh_pct, ghi_wm2, past_PV, sin_hour, cos_hour, sin_doy, cos_doy] (8次元)
Y: 未来96h の PV (1次元)
marker_y: 未来96h の ghi_wm2 (元単位, W/m^2) を 1チャネルで格納 ←★ハードゼロ用

使い方（CMDワンライナー例）:
python tools/build_dataset_96in96out_hardzero.py -w dataset_PV\weather_csv\weather_processed.csv -p dataset_PV\pv_hourly_energy_total.csv -o dataset_PV\processed_hardzero --ghi-threshold 1.0 --hist-len 96 --fut-len 96
"""
import argparse, os, json
import numpy as np
import pandas as pd
from math import sin, cos, pi

# ======================================
# 時間特徴量を追加する関数
# ======================================
def add_time_feats(df):
    # 時刻を sin/cos に変換
    h = df['time'].dt.hour.values
    doy = df['time'].dt.dayofyear.values
    sin_hour = np.sin(2 * np.pi * h / 24.0)
    cos_hour = np.cos(2 * np.pi * h / 24.0)
    sin_doy  = np.sin(2 * np.pi * (doy - 1) / 365.0)
    cos_doy  = np.cos(2 * np.pi * (doy - 1) / 365.0)
    df['sin_hour'] = sin_hour
    df['cos_hour'] = cos_hour
    df['sin_doy']  = sin_doy
    df['cos_doy']  = cos_doy
    return df

# ======================================
# メイン関数
# ======================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--weather-csv", required=True)
    ap.add_argument("-p", "--pv-csv", required=True)
    ap.add_argument("-o", "--out-dir", required=True)
    ap.add_argument("--ghi-threshold", type=float, default=1.0, help="夜間のGHIしきい値")
    ap.add_argument("--hist-len", type=int, default=96)
    ap.add_argument("--fut-len", type=int, default=96)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ============================
    # CSV読込
    # ============================
    weather = pd.read_csv(args.weather_csv)
    pv = pd.read_csv(args.pv_csv)

    # time列チェック
    if "time" not in weather.columns:
        raise ValueError("weather CSV に 'time' 列が必要です。")

    # PV側の列名を特定
    pv_time_col = "time" if "time" in pv.columns else ("datetime" if "datetime" in pv.columns else None)
    if pv_time_col is None:
        raise ValueError("pv CSV に 'time' または 'datetime' 列が必要です。")
    pv_val_col = "pv" if "pv" in pv.columns else ("pv_kWh" if "pv_kWh" in pv.columns else None)
    if pv_val_col is None:
        raise ValueError("pv CSV に 'pv' または 'pv_kWh' 列が必要です。")

    # ============================
    # datetime型に変換（タイムゾーン削除で統一）
    # ============================
    weather["time"] = pd.to_datetime(weather["time"], errors="coerce").dt.tz_localize(None)
    pv[pv_time_col] = pd.to_datetime(pv[pv_time_col], errors="coerce").dt.tz_localize(None)

    # ============================
    # マージ（時系列順で asof 結合）
    # ============================
    weather = weather.sort_values("time")
    pv = pv.sort_values(pv_time_col).rename(columns={pv_time_col: "time", pv_val_col: "pv"})
    df = pd.merge_asof(weather, pv, on="time")

    # ============================
    # 夜間ゼロ処理（GHI <= threshold のときPVを0）
    # ============================
    df.loc[df["ghi_wm2"] <= args.ghi_threshold, "pv"] = 0.0

    # ============================
    # 時間特徴量追加
    # ============================
    df = add_time_feats(df)

    # ============================
    # 欠損値除去
    # ============================
    df = df.dropna().reset_index(drop=True)

    # ============================
    # 特徴量設定
    # ============================
    feats_x = ["temp_c", "rh_pct", "ghi_wm2", "pv", "sin_hour", "cos_hour", "sin_doy", "cos_doy"]
    Xall = df[feats_x].values.astype(np.float32)
    Pall = df["pv"].values.astype(np.float32)
    Gall = df["ghi_wm2"].values.astype(np.float32)  # 未来GHI（marker_y）

    H = args.hist_len
    F = args.fut_len
    T = len(df)
    N = T - (H + F) + 1
    if N <= 0:
        raise ValueError("データ長が足りません。")

    # ============================
    # スライディングウィンドウ生成
    # ============================
    X = np.zeros((N, H, 8), np.float32)
    Y = np.zeros((N, F, 1), np.float32)
    MY = np.zeros((N, F, 1), np.float32)

    for i in range(N):
        X[i] = Xall[i:i+H]
        Y[i, :, 0] = Pall[i+H:i+H+F]
        MY[i, :, 0] = Gall[i+H:i+H+F]

    # ============================
    # スケーリング
    # ============================
    x_mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    x_std = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-8
    y_mean = Y.mean()
    y_std = Y.std() + 1e-8

    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std

    # ============================
    # データ分割（時系列 8:1:1）
    # ============================
    n_train = int(N * 0.8)
    n_val = int(N * 0.1)
    n_test = N - n_train - n_val

    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, N)

    # ============================
    # 保存（標準化済データ）
    # ============================
    np.save(os.path.join(args.out_dir, "train_x.npy"), Xn[idx_train])
    np.save(os.path.join(args.out_dir, "val_x.npy"), Xn[idx_val])
    np.save(os.path.join(args.out_dir, "test_x.npy"), Xn[idx_test])

    np.save(os.path.join(args.out_dir, "train_y.npy"), Yn[idx_train])
    np.save(os.path.join(args.out_dir, "val_y.npy"), Yn[idx_val])
    np.save(os.path.join(args.out_dir, "test_y.npy"), Yn[idx_test])

    # marker_y（未来GHIをそのまま保存）
    np.save(os.path.join(args.out_dir, "train_marker_y.npy"), MY[idx_train])
    np.save(os.path.join(args.out_dir, "val_marker_y.npy"), MY[idx_val])
    np.save(os.path.join(args.out_dir, "test_marker_y.npy"), MY[idx_test])

    # ============================
    # スケーラ保存
    # ============================
    np.savez(os.path.join(args.out_dir, "scaler_x.npz"), mean=x_mean, std=x_std)
    np.savez(os.path.join(args.out_dir, "scaler_y.npz"), mean=np.array(y_mean), std=np.array(y_std))

    # ============================
    # メタ情報保存
    # ============================
    meta = {
        "features": feats_x,
        "hist_len_h": H,
        "fut_len_h": F,
        "ghi_threshold_night": args.ghi_threshold,
        "notes": "marker_yは未来GHI(元単位)を1ch格納。hard_zeroが可能。"
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ============================
    # 完了表示
    # ============================
    print("=== Done ===")
    print("Saved to:", args.out_dir)
    print(f"Shapes: X={Xn.shape}, Y={Yn.shape}, marker_y(ghi_future)={MY.shape}")
    print("Files: train/val/test_{x,y,marker_y}.npy + scaler_x/y + meta.json")

# ======================================
# エントリーポイント
# ======================================
if __name__ == "__main__":
    main()
