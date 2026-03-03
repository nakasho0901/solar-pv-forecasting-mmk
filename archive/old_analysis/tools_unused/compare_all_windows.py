# compare_all_windows.py
# 全窓 (all test windows) に対して MMK vs iTransformer を 96時間 ahead で比較
# 入力CSVは少なくとも [window_id, h, mae] が必要 (または y_true, y_pred からmae計算)

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize_columns(df):
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["win", "window", "windowid", "window_id"]:
            rename_map[c] = "window_id"
        elif cl in ["h", "horizon", "step", "ahead"]:
            rename_map[c] = "h"
        elif cl in ["mae"]:
            rename_map[c] = "mae"
        elif cl in ["y_true", "y"]:
            rename_map[c] = "y_true"
        elif cl in ["y_pred", "pred", "forecast"]:
            rename_map[c] = "y_pred"
    df = df.rename(columns=rename_map)
    return df

def ensure_mae(df, model_name):
    if "mae" not in df.columns:
        if {"y_true", "y_pred"}.issubset(df.columns):
            df["mae"] = (df["y_true"] - df["y_pred"]).abs()
        else:
            raise ValueError(f"{model_name}: mae列 or (y_true, y_pred) が必要です")
    return df

def summarize(df):
    return {
        "mean": df["mae"].mean(),
        "median": df["mae"].median(),
        "p90": df["mae"].quantile(0.9),
        "p95": df["mae"].quantile(0.95),
        "count": len(df)
    }

def aggregate_by_h(df):
    g = df.groupby("h")["mae"]
    return pd.DataFrame({
        "h": g.mean().index,
        "mean": g.mean().values,
        "median": g.median().values,
        "q25": g.quantile(0.25).values,
        "q75": g.quantile(0.75).values
    }).sort_values("h")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmk", required=True, help="MMKの結果CSV")
    ap.add_argument("--itr", required=True, help="iTransformerの結果CSV")
    ap.add_argument("--outdir", default="allwin_results", help="出力ディレクトリ")
    ap.add_argument("--title", default="MMK vs iTransformer (96h ahead, all windows)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # --- load ---
    mmk = pd.read_csv(args.mmk)
    itr = pd.read_csv(args.itr)
    mmk = ensure_mae(normalize_columns(mmk), "MMK")
    itr = ensure_mae(normalize_columns(itr), "iTransformer")

    # horizon制限
    mmk = mmk[(mmk["h"]>=1) & (mmk["h"]<=96)]
    itr = itr[(itr["h"]>=1) & (itr["h"]<=96)]

    # --- aggregate ---
    mmk_agg = aggregate_by_h(mmk)
    itr_agg = aggregate_by_h(itr)

    # --- line plot ---
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(mmk_agg["h"], mmk_agg["mean"], label="MMK")
    ax.fill_between(mmk_agg["h"], mmk_agg["q25"], mmk_agg["q75"], alpha=0.2)
    ax.plot(itr_agg["h"], itr_agg["mean"], label="iTransformer")
    ax.fill_between(itr_agg["h"], itr_agg["q25"], itr_agg["q75"], alpha=0.2)
    ax.set_xlabel("Horizon (hours ahead)")
    ax.set_ylabel("MAE")
    ax.set_title(args.title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "mae_vs_h.png"), dpi=200)
    plt.close(fig)

    # --- boxplot (全hまとめ) ---
    fig2, ax2 = plt.subplots(figsize=(7,6))
    ax2.boxplot([mmk["mae"], itr["mae"]], labels=["MMK","iTransformer"], showfliers=False)
    ax2.set_ylabel("MAE")
    ax2.set_title("MAE distribution across all horizons")
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.outdir, "boxplot_allh.png"), dpi=200)
    plt.close(fig2)

    # --- diff plot ---
    delta = pd.DataFrame({
        "h": mmk_agg["h"],
        "delta": itr_agg["mean"] - mmk_agg["mean"]
    })
    fig3, ax3 = plt.subplots(figsize=(10,4))
    ax3.axhline(0, ls="--", c="k", lw=1)
    ax3.plot(delta["h"], delta["delta"])
    ax3.set_xlabel("Horizon (hours ahead)")
    ax3.set_ylabel("Δ mean MAE (iTr - MMK)")
    ax3.set_title("Mean MAE difference (negative=MMK better)")
    fig3.tight_layout()
    fig3.savefig(os.path.join(args.outdir, "delta_mae.png"), dpi=200)
    plt.close(fig3)

    # --- summary ---
    mmk_sum = summarize(mmk); mmk_sum["model"]="MMK"
    itr_sum = summarize(itr); itr_sum["model"]="iTransformer"
    pd.DataFrame([mmk_sum, itr_sum]).to_csv(os.path.join(args.outdir,"summary.csv"), index=False)

    # --- horizon table ---
    out = mmk_agg.merge(itr_agg, on="h", suffixes=("_mmk","_itr"))
    out.to_csv(os.path.join(args.outdir,"horizon_stats.csv"), index=False)

    print(f"[OK] 結果保存: {args.outdir}")

if __name__ == "__main__":
    main()
