#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import pandas as pd

def read_csv_auto(path):
    # Try UTF-8 with BOM first, then CP932(SJIS)
    for enc in ("utf-8-sig", "utf-8", "cp932"):
        try:
            return pd.read_csv(path, encoding=enc), enc
        except Exception:
            continue
    raise RuntimeError("Failed to read CSV with utf-8-sig/utf-8/cp932.")

def main():
    ap = argparse.ArgumentParser(description="Drop precipitation columns (含む: '降水').")
    ap.add_argument("--src", required=True, help="Input CSV path")
    ap.add_argument("--dst", required=True, help="Output CSV path")
    ap.add_argument("--dryrun", action="store_true", help="Preview only (no write)")
    args = ap.parse_args()

    df, enc = read_csv_auto(args.src)
    cols = list(df.columns)

    # 判定: 列名に「降水」を含む列を対象（例: 降水量(mm), 降水, 降水量）
    drop_cols = [c for c in cols if "降水" in str(c)]
    if not drop_cols:
        print("⚠ 降水系の列が見つかりませんでした。保持列:", ", ".join(cols))
        if not args.dryrun:
            # そのままコピー保存
            df.to_csv(args.dst, index=False, encoding="utf-8-sig")
            print(f"→ 変更なしで保存: {args.dst}")
        sys.exit(0)

    print("削除対象の列:", ", ".join(drop_cols))
    kept_cols = [c for c in cols if c not in drop_cols]
    print("保持する列:", ", ".join(kept_cols))

    if args.dryrun:
        print("Dry-runのため書き込みしません。")
        sys.exit(0)

    df = df.drop(columns=drop_cols)
    # ヘッダーの文字化け回避で utf-8-sig で保存
    df.to_csv(args.dst, index=False, encoding="utf-8-sig")
    print(f"[OK] 保存しました: {args.dst} （元エンコーディング: {enc}）")
    print(f"列数: {len(kept_cols)} / 行数: {len(df)}")

if __name__ == "__main__":
    main()
