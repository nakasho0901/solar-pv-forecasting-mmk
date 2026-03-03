# make_tsukuba_npz.py
import sys, os, glob
import numpy as np
import pandas as pd

# ==== 1) 入力設定（必要に応じて書き換え）===========================
# A. 既に「時刻」「PV出力」列を持つ単一CSVがある場合（推奨）
HOURLY_CSV = None  # 例: r"dataset_PV\output\pv_hourly.csv"（あればパスを書く）

# B. 複数CSVや秒データしかない場合：このフォルダ配下を再帰的に探索して自動統合
SRC_DIR = r"dataset_PV"      # 画像のフォルダ（dataset_PV）を指定
FREQ_MIN = 60                # 学習に使うサンプリング間隔（分）→ 60なら1時間

# 列名の候補（自動検出用・足りなければ追加）
TIME_COL_CANDS = ["time", "timestamp", "datetime", "date", "時刻", "日時"]
PV_COL_CANDS   = ["pv", "pv_kw", "pv_power", "solar_power", "発電", "発電量", "出力", "P_pv", "kW"]

# ==== 2) ユーティリティ =============================================
def read_csv_any(fp):
    for enc in ["utf-8-sig", "cp932", "utf-16"]:
        try:
            return pd.read_csv(fp, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(fp)  # 最後はデフォルト

def find_col(df, cands):
    cols = [c for c in df.columns]
    low = {c.lower(): c for c in cols}
    # 完全一致（小文字）
    for key in cands:
        if key.lower() in low:
            return low[key.lower()]
    # 部分一致
    for c in cols:
        cl = c.lower()
        if any(key.lower() in cl for key in cands):
            return c
    return None

def to_hourly(df, time_col, val_col):
    df = df[[time_col, val_col]].copy()
    # 日時へ
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])
    df = df.set_index(time_col).sort_index()

    # 値をfloatへ
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df = df.dropna(subset=[val_col])

    # 秒/分データ→時間平均（「出力[kW]」として扱う想定）
    rule = f"{FREQ_MIN}T"
    hourly = df.resample(rule).mean()

    # 物理的にありえない負値は0に丸める（任意）
    hourly[val_col] = hourly[val_col].clip(lower=0)

    # 短い欠損は時間補間（任意）
    hourly[val_col] = hourly[val_col].interpolate("time", limit=3)

    return hourly

# ==== 3) データの読み込み ===========================================
if HOURLY_CSV and os.path.exists(HOURLY_CSV):
    df = read_csv_any(HOURLY_CSV)
    tcol = find_col(df, TIME_COL_CANDS)
    pcol = find_col(df, PV_COL_CANDS)
    if tcol is None or pcol is None:
        print("列検出に失敗: time/PV 列名をスクリプト先頭で指定してください。")
        sys.exit(1)
    hourly = to_hourly(df, tcol, pcol)
else:
    # 自動探索：フォルダ配下のCSVを全部なめて、PV列があるものだけ拾う
    csvs = glob.glob(os.path.join(SRC_DIR, "**", "*.csv"), recursive=True)
    if not csvs:
        print("CSVが見つかりませんでした。SRC_DIR を確認してください。")
        sys.exit(1)

    pieces = []
    for fp in csvs:
        try:
            tmp = read_csv_any(fp)
            tcol = find_col(tmp, TIME_COL_CANDS)
            pcol = find_col(tmp, PV_COL_CANDS)
            if tcol and pcol:
                h = to_hourly(tmp, tcol, pcol)
                h = h.rename(columns={pcol: "pv"})
                pieces.append(h[["pv"]])
        except Exception:
            continue

    if not pieces:
        print("PV列を含むCSVが見つかりません。PV_COL_CANDS を増やすか HOURLY_CSV を指定してください。")
        sys.exit(1)

    # 重複があれば平均（同じ時間に複数ファイルの値がある場合）
    hourly = pd.concat(pieces).groupby(level=0).mean().sort_index()

# ==== 4) npzへ保存 =================================================
# 最終クリーニング
hourly = hourly.dropna()
hourly = hourly[~hourly.index.duplicated(keep="first")]

# 学習用配列へ
variable = hourly.iloc[:, [0]].to_numpy(dtype=np.float32)   # (T, 1)
timestamp = hourly.index.values.astype("datetime64[ns]")  # ← 文字列ではなく datetime64 で保存

mean = variable.mean(axis=0)
std  = variable.std(axis=0) + 1e-8

# 出力先
os.makedirs("dataset", exist_ok=True)
out = os.path.join("dataset", "Tsukuba.npz")
np.savez(out, variable=variable, timestamp=timestamp, mean=mean, std=std)

# 参考出力
T = variable.shape[0]
print(f"Saved: {out}")
print(f"Length T={T}, var_num=1, freq={FREQ_MIN} min")
print("Head:")
print(hourly.head(3))
