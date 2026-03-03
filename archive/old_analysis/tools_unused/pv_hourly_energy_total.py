# pv_hourly_energy_total.py
from pathlib import Path
import pandas as pd
import re

# === あなたの環境に合わせてください ===
DATA_DIR = r"C:\Users\nakas\my_project\solar-kan\dataset_PV\raw_sec"

# ---------- ユーティリティ ----------
def clean_datetime_str(s: pd.Series) -> pd.Series:
    """
    '2015/01/01 00:00:00 のような先頭の余計な記号を除去し、
    %Y/%m/%d %H:%M:%S 形式に合わせやすくする
    """
    return s.astype(str).str.replace(r"[^\d/: \-T]", "", regex=True).str.strip()

# ---------- データ読み込み＆処理 ----------
def load_and_extract_pv(path: Path) -> pd.DataFrame:
    """
    1ファイルから日時と PV(合算kW) を抽出して返す
    - 先頭3行は: 日本語列名 / センサーID / 単位
    - 実データは4行目以降
    """
    # 実データ部
    df = pd.read_csv(path, encoding="cp932", skiprows=3, header=None)
    df = df.rename(columns={0: "datetime"})

    # 日本語の列名（1行目）を取得して PV 列のインデックスを特定
    header = pd.read_csv(path, encoding="cp932", nrows=1, header=None)
    colnames = list(header.iloc[0])

    pv_col_idx = []
    for i, name in enumerate(colnames):
        if isinstance(name, str) and ("太陽光発電" in name) and ("有効電力" in name):
            pv_col_idx.append(i)

    if not pv_col_idx:
        print(f"[WARN] {path.name} : PV列が見つかりませんでした")
        return pd.DataFrame(columns=["datetime", "pv_kw"])

    # PV列を数値化して合算
    pv_df = df[pv_col_idx].apply(pd.to_numeric, errors="coerce")
    df["pv_kw"] = pv_df.sum(axis=1)

    # 日時を高速・安定にパース
    dt_str = clean_datetime_str(df["datetime"])
    dt = pd.to_datetime(dt_str, format="%Y/%m/%d %H:%M:%S", errors="coerce")
    # 万一フォーマットが違っていた場合はフォールバック
    if dt.notna().sum() < 10:
        dt = pd.to_datetime(dt_str, errors="coerce")
    df["datetime"] = dt

    return df[["datetime", "pv_kw"]]

# ---------- メイン処理 ----------
def main():
    data_path = Path(DATA_DIR)
    files = [p for p in sorted(data_path.iterdir()) if p.is_file()]
    print(f"検出ファイル数: {len(files)}")

    dfs = []
    for p in files:
        try:
            df = load_and_extract_pv(p)
            valid = df["datetime"].notna().sum()
            if valid < 10:
                print(f"[SKIP] {p.name} : datetime不足 ({valid})")
                continue
            print(f"[OK]   {p.name} : {len(df)} 行")
            dfs.append(df)
        except Exception as e:
            print(f"[ERROR] {p.name} : {e}")

    if not dfs:
        print("PVデータが取得できませんでした")
        return

    # 結合・整形
    raw = pd.concat(dfs, ignore_index=True)
    raw = (
        raw.dropna(subset=["datetime"])
           .sort_values("datetime")
           .drop_duplicates(subset=["datetime"])
           .set_index("datetime")
    )

    # ---- 重要：負値は0にクリップ（夜間のノイズ対策）----
    raw["pv_kw"] = raw["pv_kw"].clip(lower=0)

    # 1秒[kW] → 1時間[kWh] に変換（積算）
    raw["pv_kWh"] = raw["pv_kw"] / 3600.0
    hourly = raw["pv_kWh"].resample("1h").sum().to_frame()

    # 保存
    out_dir = data_path.parent / "prepared"
    out_dir.mkdir(exist_ok=True)
    outfile = out_dir / "pv_hourly_energy_total.csv"
    hourly.reset_index().to_csv(outfile, index=False)

    print("=== 完了 ===")
    print("保存先:", outfile)
    print("1時間テーブル 行数:", len(hourly))
    print(hourly.head())

if __name__ == "__main__":
    main()
