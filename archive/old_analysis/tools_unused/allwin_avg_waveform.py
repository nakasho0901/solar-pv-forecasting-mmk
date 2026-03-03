# allwin_avg_waveform.py
# 全窓を集計して「平均波形」を描くスクリプト

import pandas as pd
import matplotlib.pyplot as plt

# ▼あなたの評価CSVファイルに合わせて変更してください
mmk_csv = "results/MMK_preds.csv"
itr_csv = "results/iTransformer_preds.csv"

# --- 読み込み ---
mmk = pd.read_csv(mmk_csv)
itr = pd.read_csv(itr_csv)

# 列名を揃える（window_id, h, y_true, y_pred があればOK）
mmk = mmk.rename(columns={"win":"window_id","window":"window_id","step":"h","ahead":"h",
                          "pred":"y_pred","forecast":"y_pred"})
itr = itr.rename(columns={"win":"window_id","window":"window_id","step":"h","ahead":"h",
                          "pred":"y_pred","forecast":"y_pred"})

# horizonを1..96に制限
mmk = mmk[(mmk["h"]>=1) & (mmk["h"]<=96)]
itr = itr[(itr["h"]>=1) & (itr["h"]<=96)]

# --- 全窓平均 ---
mmk_mean = mmk.groupby("h")[["y_true","y_pred"]].mean().reset_index()
itr_mean = itr.groupby("h")["y_pred"].mean().reset_index()

# --- プロット ---
plt.figure(figsize=(12,5))
plt.plot(mmk_mean["h"], mmk_mean["y_true"], label="True Values", color="blue")
plt.plot(mmk_mean["h"], mmk_mean["y_pred"], "--", label="MMK (Pred)", color="orange")
plt.plot(itr_mean["h"], itr_mean["y_pred"], "--", label="iTransformer (Pred)", color="green")

plt.title("Average across all windows (96h ahead)")
plt.xlabel("Time Steps (Hours)")
plt.ylabel("Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("allwin_avg_waveform.png", dpi=200)
plt.show()
