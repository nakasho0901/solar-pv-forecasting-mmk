# tools/make_tsukuba_from_pv_only.py
import argparse, os
import numpy as np
import pandas as pd

CAND_TS = ["datetime","timestamp","time","date","日時","時刻","観測時刻"]

def read_csv_smart(path):
    for enc in ["utf-8", "utf-8-sig", "cp932", "shift_jis"]:
        try:
            return pd.read_csv(path, encoding=enc, engine="python")
        except Exception:
            pass
    return pd.read_csv(path, engine="python")

def find_timestamp_col(df):
    lower = {str(c).lower(): c for c in df.columns}
    for k in CAND_TS:
        if k in lower: 
            return lower[k]
    # 先頭列が日時っぽければ採用
    c0 = df.columns[0]
    try:
        pd.to_datetime(df[c0].head(10), errors="raise")
        return c0
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="pv_hourly_energy_total.csv のパス")
    ap.add_argument("--out", default="dataset/Tsukuba.npz", help="出力 npz のパス")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = read_csv_smart(args.csv)
    ts_col = find_timestamp_col(df)
    if ts_col is None:
        raise ValueError("タイムスタンプ列を特定できません。'datetime' や 'timestamp' 列を用意してください。")

    df = df.copy()
    df["__ts__"] = pd.to_datetime(df[ts_col], errors="raise")
    df = df.dropna(subset=["__ts__"]).sort_values("__ts__").reset_index(drop=True)

    # 数値列だけを採用（タイムスタンプ列は除外）
    num_cols = [c for c in df.columns if c != "__ts__"]
    num_df = df[num_cols].apply(pd.to_numeric, errors="coerce")
    num_df = num_df.loc[:, num_df.notna().any()]  # 全NaN列は除去
    if num_df.shape[1] == 0:
        raise ValueError("数値列が見つかりません。pvの列名を数値として読めるようにしてください。")

    variable = num_df.astype("float32").to_numpy()                  # (T, N)
    timestamp = df["__ts__"].astype("datetime64[ns]").astype(str)   # (T,)

    mean = np.nanmean(variable, axis=0).astype("float32")
    std  = (np.nanstd(variable, axis=0).astype("float32") + 1e-8)

    np.savez_compressed(args.out, variable=variable, timestamp=timestamp, mean=mean, std=std)
    print(f"saved: {args.out}")
    print(f"shape: T={variable.shape[0]}, N={variable.shape[1]}")

if __name__ == "__main__":
    main()
