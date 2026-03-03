#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd

def read_csv_auto(path):
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc), enc
        except Exception:
            continue
    raise RuntimeError("Failed to read CSV with utf-8-sig/utf-8/cp932.")

def main():
    ap = argparse.ArgumentParser(description="日本語見出しの気象CSVを正規化して add_weather_features.py に渡せる形にします。")
    ap.add_argument("--src", required=True, help="入力CSV（日本語ヘッダー）")
    ap.add_argument("--dst", required=True, help="出力CSV（英語・標準化ヘッダー）")
    ap.add_argument("--tz", default="Asia/Tokyo")
    args = ap.parse_args()

    df, enc = read_csv_auto(args.src)

    # 列名の正規化（存在すればマップ）
    colmap = {}
    # 時刻
    if "年月日時" in df.columns:
        colmap["年月日時"] = "time"
    elif "日時" in df.columns:
        colmap["日時"] = "time"
    # 気温
    for cand in ["気温(℃)", "気温", "外気温(℃)"]:
        if cand in df.columns:
            colmap[cand] = "temp_c"
            break
    # 相対湿度（% / ％ どちらも想定）
    for cand in ["相対湿度(%)", "相対湿度(％)", "相対湿度"]:
        if cand in df.columns:
            colmap[cand] = "rh_pct"
            break
    # 日射量（MJ/㎡）
    for cand in ["日射量(MJ/㎡)", "日射量(MJ/m2)", "全天日射量(MJ/㎡)"]:
        if cand in df.columns:
            colmap[cand] = "ghi_mj_m2"      # まずは MJ/m2 として受ける
            break

    if "time" not in colmap.values():
        raise ValueError("時刻列（例: 年月日時）が見つかりません。")

    df = df.rename(columns=colmap)

    # 時刻をパース（タイムゾーンは Asia/Tokyo を付与）
    df["time"] = pd.to_datetime(df["time"], errors="coerce", infer_datetime_format=True)
    if df["time"].isna().any():
        bad = df["time"].isna().sum()
        raise ValueError(f"時刻が解釈できない行があります: {bad} 行")
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize(args.tz)
    else:
        df["time"] = df["time"].dt.tz_convert(args.tz)
    df["time"] = df["time"].dt.floor("H")

    # GHI 変換: MJ/m2（1時間値）→ W/m2（平均放射照度）
    # 1 MJ/m2 / 1h = (1e6 J/m2) / 3600 s ≈ 277.777... W/m2
    if "ghi_mj_m2" in df.columns:
        df["ghi_wm2"] = df["ghi_mj_m2"].astype(float) * (1_000_000.0 / 3600.0)
        # 値が明らかに異常な場合（例: >1500 W/m2）は NaN に
        df.loc[df["ghi_wm2"] > 1500, "ghi_wm2"] = pd.NA

    # 出力に含める列（存在するものだけ）
    keep = ["time", "temp_c", "rh_pct", "ghi_wm2"]
    out = [c for c in keep if c in df.columns]
    df = df[out].sort_values("time").reset_index(drop=True)

    df.to_csv(args.dst, index=False, encoding="utf-8-sig")
    print(f"[OK] 正規化完了: {args.dst}  (元エンコーディング: {enc})")
    print("列:", ", ".join(df.columns))
    print("行:", len(df))

if __name__ == "__main__":
    main()
