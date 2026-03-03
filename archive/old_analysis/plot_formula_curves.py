# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_legendre # Jacobiの近似として使用

# 先ほど抽出した係数（Layer 0）
taylor_coeffs = [0.0044, 0.0038, -0.0040, 0.0009]
jacobi_coeffs = [-0.0035, -0.0006, -0.0020, 0.0018, 0.0007]

# 入力範囲（正規化された -1 から 1）
x = np.linspace(-1, 1, 100)

# 数式の計算
y_taylor = sum(c * (x**i) for i, c in enumerate(taylor_coeffs))
y_jacobi = sum(c * eval_legendre(i, x) for i, c in enumerate(jacobi_coeffs))

# 描画設定
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Taylor (Power担当)
ax[0].plot(x, y_taylor, label='Taylor (Layer 0)', color='red', lw=3)
ax[0].set_title("Learned Physics: Power Prediction (Taylor)")
ax[0].set_xlabel("Normalized Input (e.g. Solar Radiation)")
ax[0].set_ylabel("Effect on Output")
ax[0].grid(True, alpha=0.3)
ax[0].legend()

# Jacobi (Temp担当)
ax[1].plot(x, y_jacobi, label='Jacobi (Layer 0)', color='blue', lw=3)
ax[1].set_title("Learned Physics: Temp Prediction (Jacobi)")
ax[1].set_xlabel("Normalized Input (e.g. Temperature)")
ax[1].set_ylabel("Effect on Output")
ax[1].grid(True, alpha=0.3)
ax[1].legend()

plt.tight_layout()
plt.savefig("kan_learned_curves.png", dpi=300)
print("【成功】グラフを保存しました: kan_learned_curves.png")
plt.show()