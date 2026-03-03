# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import os

# [cite_start]1. パラメータ設定 (解析結果より) [cite: 778-782, 786]
c0, c1, c2, c3 = 0.0044, 0.0038, -0.0040, 0.0009
scale = 33.01  # kWh換算用のスケール
mean = 20.54   # kWh換算用の平均

# 2. テストデータの読み込み
data_path = "dataset_PV/processed_hardzero/test_x.npy"
if not os.path.exists(data_path):
    # データがない場合はシミュレーション用のダミーデータを作成
    print("【通知】実データが見つからないため、シミュレーション用サンプルを生成します。")
    solar_input = np.linspace(0, 1, 96) 
else:
    test_x = np.load(data_path)
    # [cite_start]日射量(Var 2)の1サンプル(96時間分)を抽出 [cite: 345-349]
    solar_input = test_x[10, :, 2] 

# 3. 数式シミュレーション
# 線形部分のみ
y_linear = (c0 + c1 * solar_input) * scale + mean
# 非線形（補正）部分
y_correction = (c2 * (solar_input**2) + c3 * (solar_input**3)) * scale
# 合計 (Full予測)
y_full = y_linear + y_correction

# 4. 描画
plt.figure(figsize=(12, 6))

# メイングラフ
plt.plot(y_linear, 'b--', label='Linear Prediction (No Non-linear terms)', alpha=0.7)
plt.plot(y_full, 'r', label='Full MMK Prediction (with $x^2, x^3$)', lw=2)

# 補正（仕事量）を塗りつぶし
plt.fill_between(range(len(solar_input)), y_linear, y_full, color='orange', alpha=0.3, label='Non-linear Correction (Safety Brake)')

plt.title("Simulation: How Non-linear Terms Suppress Overestimation")
plt.xlabel("Time Steps (Hours)")
plt.ylabel("Power Generation [kWh]")
plt.legend()
plt.grid(True, alpha=0.3)

# 統計情報の表示
max_correction = np.max(np.abs(y_correction))
print(f"--- シミュレーション結果 ---")
print(f"最大補正量（ブレーキ量）: {max_correction:.4f} kWh")
print(f"ピーク時の予測抑制率: {max_correction / np.max(y_full) * 100:.2f} %")

plt.savefig("simulation_result_kwh.png", dpi=300)
print("【完了】シミュレーション図を保存しました: simulation_result_kwh.png")
plt.show()