# -*- coding: utf-8 -*-
# 最後のウィンドウ（テストデータの最終96時間）を可視化するスクリプト

import numpy as np
import matplotlib.pyplot as plt

# ===== データ読込 =====
Y = np.load('dataset_PV/splits_96_96_weather/test_y.npy')
pred_iT = np.load('save/iTransformer_Tsukuba/preds/test_preds.npy')
pred_MMK = np.load('save/MMK_Tsukuba/preds/test_preds.npy')

# ===== 最後のウィンドウを選択 =====
window_idx = len(Y) - 1
print(f"Plotting the LAST window (index={window_idx})")

y_true = Y[window_idx].squeeze()
y_pred_iT = pred_iT[window_idx].squeeze()
y_pred_MMK = pred_MMK[window_idx].squeeze()

# ===== グラフ描画 =====
plt.figure(figsize=(10, 5))
plt.plot(y_true, label='True (Ground Truth)', color='black', linewidth=2)
plt.plot(y_pred_iT, label='iTransformer', color='blue', linestyle='--')
plt.plot(y_pred_MMK, label='MMK', color='red', linestyle=':')
plt.title(f'PV Forecast Comparison (Last Window: {window_idx})')
plt.xlabel('Time step (96 steps ahead)')
plt.ylabel('PV Power [same unit as test_y.npy]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
