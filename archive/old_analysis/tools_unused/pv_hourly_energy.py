# pv_hourly_energy_split.py
from pathlib import Path
import pandas as pd
import re

# === あなたの環境に合わせる ===
DATA_DIR = r"C:\Users\nakas\my_project\solar-kan\dataset_PV\raw_sec"

# ---------- ユーティリティ ----------
def read_head_text(p: Path, nbytes=512) -> str:
    with p.open("rb") as f:
        return f.read(nbytes).decode("utf-8", errors="ignore")

def detect_sep(path: Path, default=","):
    head = read_head_text(path)
    return ";" if head.count(";") > head.count(",") else default

def looks_like_header(path: Path) -> bool:
    first = read_head_text(path, 256).splitlines()[0] if path.exists() else ""
    return any(ch.isalpha() or '\u3040' <= ch <= '\u9fff' for ch in first)

def normalize_colname(x) -> str:
    s = str(x).replace("\u3000", " ")
    return re.sub(r"\s+", " ", s).strip()

def num_datetime(s: pd.Series) -> pd.Series:
    """数値や文字で与えられた yyyymmdd / hhmmss を datetime に変換"""
    s2 = s.astype(str).str.replace(r"\D", "", regex=True)
    if s2.str.len().eq(14).any():
        return pd.to_datetime(s2, format="%Y%m%d%H%M%S", errors="coerce")
    if s2.str.len().eq(8).any():
        return pd.to_datetime(s2, format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s2, errors="coerce")

def parse_datetime(df: pd.DataFrame) -> pd.Series:
    """datetime列を生成（date+time 両方に対応）"""
    cols = {c: normalize_colname(c).lower() for c in df.columns}

    # 日付＋時刻 列がある場合
    date_candidates = [c for c, cl in cols.items() if "date" in cl or "日付" in cl or "年月日" in cl]
    time_candidates = [c for c, cl in cols.items() if "time" in cl or "時刻" in cl or "時間" in cl]

    if date_candidates and time_candidates:
        d = df[date_candidates[0]].astype(str).str.replace(r"\D", "", regex=True)
        t = df[time_candidates[0]].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
        dt = pd.to_datetime(d + t, format="%Y%m%d%H%M%S", errors="coerce")
        return dt

    # 単一列 datetime の場合
    for c, cl in cols.items():
        if "time" in cl or "日時" in cl or "date" in cl:
            return pd.to_datetime(df[c], errors="coerce")

    # フォールバック: 先頭列
    return pd.to_datetime(df.iloc[:,0], errors="coerce")

def extract_pv_split(df: pd.DataFrame) -> pd.DataFrame:
    """PV1〜4列を個別に抽出して数値化"""
    df = df.copy()
    df.columns = [normalize_colname(c) for c in df.columns]
    df["datetime"] = parse_datetime(df)

    pv_map = {}
    for c in df.columns:
        if re.search(r"(太陽光|PV).*1.*(有効電力|出力|kW|power)", c, flags=re.I):
            pv_map[c] = "pv1_kw"
        elif re.search(r"(太陽光|PV).*2.*(有効電力|出力|kW|power)", c, flags=re.I):
            pv_map[c] = "pv2_kw"
        elif re.search(r"(太陽光|PV).*3.*(有効電力|出力|kW|power)", c, flags=re.I):
            pv_map[c] = "pv3_kw"
        elif re.search(r"(太陽光|PV).*4.*(有効電力|出力|kW|power)", c, flags=re.I):
            pv_map[c] = "pv4_kw"

    df = df.rename(columns=pv_map)
    for c in pv_map.values():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    keep = ["datetime"] + list(pv_map.values())
    return df[keep]

# ---------- メイン ----------
data_path = Path(DATA_DIR)
files = [p for p in sorted(data_path.iterdir())
         if p.is_file() and (p.suffix.lower() in (".csv",".txt") or (p.suffix=="" and p.name.lower().endswith("seccsv")))]
print(f"検出ファイル数: {len(files)}")

dfs = []
for p in files:
    sep = detect_sep(p)
    header = 0 if looks_like_header(p) else None
    try:
        df0 = pd.read_csv(p, sep=sep, encoding="cp932", header=header, dtype=str)
    except UnicodeDecodeError:
        df0 = pd.read_csv(p, sep=sep, encoding="utf-8", header=header, dtype=str)

    df = extract_pv_split(df0)
    valid = df["datetime"].notna().sum()
    if valid < 10:
        print(f"[SKIP] {p.name} : datetimeが不十分")
        continue
    print(f"[OK]   {p.name} : 有効行数 {valid}")
    dfs.append(df)

if not dfs:
    raise RuntimeError("有効な発電データが見つかりませんでした。")

raw = pd.concat(dfs, ignore_index=True)
raw = raw.dropna(subset=["datetime"]).sort_values("datetime").drop_duplicates(subset=["datetime"])
raw = raw.set_index("datetime")

# --- 1秒[kW] → 1時間[kWh] に変換 ---
hourly = pd.DataFrame(index=raw.index)
for c in ["pv1_kw","pv2_kw","pv3_kw","pv4_kw"]:
    if c in raw.columns:
        hourly[c.replace("_kw","_kWh")] = (raw[c] / 3600.0).resample("1h").sum()

# 保存
out_dir = data_path.parent / "prepared"
out_dir.mkdir(exist_ok=True)
hourly.reset_index().to_csv(out_dir / "pv_hourly_energy_split.csv", index=False)

print("=== 完了 ===")
print("保存先:", out_dir)
print("1時間テーブル 行数:", len(hourly))
print(hourly.head())
