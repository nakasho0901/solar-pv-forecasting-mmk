#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from typing import Dict, List
import numpy as np
import pandas as pd

# 英日どちらでも拾えるマッピング（気象）
CANONICALS_WEATHER = {
    "temp_c":   ["temp_c","temperature","air_temperature","気温","気温(℃)"],
    "rh_pct":   ["rh_pct","humidity","相対湿度","相対湿度(%)","相対湿度(％)"],
    "ghi_wm2":  ["ghi_wm2","solar_radiation","ghi","日射","全天日射","日射量(W/㎡)"],
    "cloud_pct":["cloud_pct","cloud_cover","雲量","雲量(%)","雲量(％)"],
    "wind_ms":  ["wind_ms","wind_speed","風速","風速(m/s)"],
    "press_hpa":["press_hpa","pressure","気圧","気圧(hPa)"],
    "rain_mm":  ["rain_mm","precip","precipitation","降水量","降水量(mm)"],
}

AGG_MEAN = ["temp_c","rh_pct","wind_ms","cloud_pct","ghi_wm2","press_hpa"]
AGG_SUM  = ["rain_mm"]

# 時刻候補（PV/気象）
TIME_CANDIDATES = ["time","timestamp","datetime","日時","年月日時","date"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pv_csv", required=True)
    p.add_argument("--weather_csv", required=True)
    p.add_argument("--out_csv", required=True)
    p.add_argument("--tz", default="Asia/Tokyo")
    p.add_argument("--lag_hours", nargs="*", default=[1,2,3,24], type=int)
    p.add_argument("--rolling_hours", default=24, type=int)
    p.add_argument("--join", default="inner", choices=["inner","left"])
    p.add_argument("--dropna_after_features", action="store_true")
    return p.parse_args()

def _find_and_rename_time(df: pd.DataFrame) -> pd.DataFrame:
    for cand in TIME_CANDIDATES:
        for col in df.columns:
            if col.lower() == cand.lower():
                return df.rename(columns={col: "time"})
    raise ValueError(f"時刻列が見つかりません（候補: {TIME_CANDIDATES}）")

def _coerce_time(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        bad = int(df["time"].isna().sum())
        raise ValueError(f"時刻が解釈できない行があります: {bad} 行")
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize(tz)
    else:
        df["time"] = df["time"].dt.tz_convert(tz)
    df["time"] = df["time"].dt.floor("h")
    return df

def _map_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for canon, alts in CANONICALS_WEATHER.items():
        for a in alts:
            for col in df.columns:
                if col.lower() == a.lower():
                    rename_map[col] = canon
                    break
    return df.rename(columns=rename_map)

def _resample_weather_hourly(dfw: pd.DataFrame, tz: str) -> pd.DataFrame:
    dfw = dfw.set_index("time").sort_index()
    agg = {}
    for c in AGG_MEAN:
        if c in dfw.columns: agg[c] = "mean"
    for c in AGG_SUM:
        if c in dfw.columns: agg[c] = "sum"
    if not agg:
        raise ValueError("集計可能な気象列が見つかりません（temp_c, rh_pct, ghi_wm2 等）")
    out = dfw.resample("1h").agg(agg).reset_index()
    out["time"] = out["time"].dt.tz_convert(tz).dt.floor("h")
    return out

def _engineer_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["time"].dt.hour
    df["dow"]  = df["time"].dt.dayofweek
    df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24.0)
    df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24.0)
    df["dow_sin"]  = np.sin(2*np.pi*df["dow"]/7.0)
    df["dow_cos"]  = np.cos(2*np.pi*df["dow"]/7.0)
    return df.drop(columns=["hour","dow"])

def _engineer_daylight(df: pd.DataFrame) -> pd.DataFrame:
    if "ghi_wm2" in df.columns:
        df["is_daylight"] = (df["ghi_wm2"] > 20).astype(int)
    else:
        h = df["time"].dt.hour
        df["is_daylight"] = ((h>=6)&(h<18)).astype(int)
    return df

def _engineer_lags_rollings(df: pd.DataFrame, lag_hours: List[int], rolling_hours: int) -> pd.DataFrame:
    df = df.sort_values("time").reset_index(drop=True)
    for col in ["pv_kwh","ghi_wm2","temp_c","cloud_pct"]:
        if col in df.columns:
            for L in lag_hours:
                df[f"lag_{col}_{L}h"] = df[col].shift(L)
    for col in ["ghi_wm2","temp_c","cloud_pct"]:
        if col in df.columns:
            df[f"roll_{col}_{rolling_hours}h_mean"] = df[col].rolling(window=rolling_hours, min_periods=max(1, rolling_hours//3)).mean()
    return df

def main():
    args = parse_args()

    # --- PV ---
    dfp = pd.read_csv(args.pv_csv)
    dfp = _find_and_rename_time(dfp)      # datetime など -> time
    dfp = _coerce_time(dfp, args.tz)
    if "pv_kwh" not in dfp.columns:
        # pv_kWh / PV / energy_kwh 等を拾って pv_kwh に統一
        alt = None
        for c in dfp.columns:
            if c.lower() in ("pv_kwh","pv_kwh ","pv","pv_energy","energy_kwh","kwh","pv_kwh".lower(),"pvkwh","pv_kw","pv_kw_h"):
                alt = c; break
        if alt is None:
            # 大文字小文字を無視して「pv」を含めば採用（最頻）
            cand = [c for c in dfp.columns if "pv" in c.lower()]
            if cand: alt = cand[0]
        if alt is None:
            raise ValueError(f"PV列が見つかりません。列名一覧: {list(dfp.columns)}")
        dfp = dfp.rename(columns={alt: "pv_kwh"})
    dfp = dfp[["time","pv_kwh"]]

    # --- Weather ---
    dfw = pd.read_csv(args.weather_csv)
    dfw = _find_and_rename_time(dfw)
    dfw = _coerce_time(dfw, args.tz)
    dfw = _map_weather_columns(dfw)
    dfwh = _resample_weather_hourly(dfw, args.tz)

    # --- Merge & Features ---
    df = pd.merge(dfp, dfwh, on="time", how=args.join)
    df = _engineer_calendar(df)
    df = _engineer_daylight(df)
    df = _engineer_lags_rollings(df, args.lag_hours, args.rolling_hours)

    if args.dropna_after_features:
        df = df.dropna().reset_index(drop=True)

    df = df.sort_values("time")
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote: {args.out_csv}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("Columns:", ", ".join(df.columns))

if __name__ == "__main__":
    main()
