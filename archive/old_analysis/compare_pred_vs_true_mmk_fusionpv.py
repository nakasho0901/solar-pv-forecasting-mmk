# ============================================================
# MMK_Mix_FusionPV : Prediction vs Ground Truth (Test set)
# ============================================================
# - last.ckpt をロードして test_x から予測
# - test_y と比較（96-96）
# - MAE / MSE / RMSE を計算
# - 時系列プロットを保存
# ============================================================

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# =========================
# パス設定（あなたの環境）
# =========================
CKPT_PATH = r"save\MMK_Mix_FusionPV\version_1\checkpoints\last.ckpt"
DATA_DIR  = r"dataset_PV\prepared\96-96_pvin_noscale_noday"
CONFIG_PY = r"config\tsukuba_conf\MMK_Mix_FusionPV_RevIN_PV_96.py"

OUT_DIR = "results_pred_vs_true"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# データ読み込み
# =========================
test_x = np.load(os.path.join(DATA_DIR, "test_x.npy"))      # (B, 96, N)
test_y = np.load(os.path.join(DATA_DIR, "test_y.npy"))      # (B, 96, 1)
marker_y = np.load(os.path.join(DATA_DIR, "test_marker_y.npy"))  # (B, 96)

print("[INFO] test_x shape :", test_x.shape)
print("[INFO] test_y shape :", test_y.shape)

# =========================
# config を python として読む
# =========================
exp_conf = {}
with open(CONFIG_PY, "r", encoding="utf-8") as f:
    exec(f.read(), exp_conf)

conf = exp_conf["exp_conf"]

# =========================
# モデル定義
# =========================
from easytsf.model.MMK_Mix_FusionPV import MMK_Mix_FusionPV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MMK_Mix_FusionPV(
    hist_len   = conf["hist_len"],
    pred_len   = conf["pred_len"],
    var_num    = conf["var_num"],
    hidden_dim = conf["hidden_dim"],
    layer_hp   = conf["layer_hp"],
    layer_num  = conf["layer_num"],
    enforce_nonneg = True,   # ★ 推論時はON推奨
).to(device)

# =========================
# checkpoint 読み込み
# =========================
ckpt = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(ckpt["state_dict"], strict=False)
model.eval()

# =========================
# 推論
# =========================
with torch.no_grad():
    x = torch.tensor(test_x, dtype=torch.float32).to(device)
    pred = model(x)   # (B, 96, 1)

pred = pred.cpu().numpy()

# =========================
# reshape（比較用）
# =========================
y_true = test_y.reshape(-1)   # (B*96,)
y_pred = pred.reshape(-1)
marker = marker_y.reshape(-1)

# =========================
# 評価指標（全体）
# =========================
mae  = mean_absolute_error(y_true, y_pred)
mse  = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)

print("===== ALL TIME =====")
print(f"MAE  : {mae:.3f} kW")
print(f"MSE  : {mse:.3f} (kW)^2")
print(f"RMSE : {rmse:.3f} kW")

# =========================
# 日中のみ評価（marker_y == 1）
# =========================
mask = marker == 1
mae_d  = mean_absolute_error(y_true[mask], y_pred[mask])
mse_d  = mean_squared_error(y_true[mask], y_pred[mask])
rmse_d = np.sqrt(mse_d)

print("===== DAYLIGHT ONLY =====")
print(f"MAE  : {mae_d:.3f} kW")
print(f"RMSE : {rmse_d:.3f} kW")

# =========================
# プロット（先頭 N 時間）
# =========================
N_PLOT = 500

plt.figure(figsize=(14, 4))
plt.plot(y_true[:N_PLOT], label="True", linewidth=2)
plt.plot(y_pred[:N_PLOT], label="Pred", linestyle="--")
plt.xlabel("Time step")
plt.ylabel("PV output [kW]")
plt.title("MMK_Mix_FusionPV : Prediction vs Ground Truth (Test)")
plt.legend()
plt.tight_layout()

out_path = os.path.join(OUT_DIR, "pred_vs_true_test.png")
plt.savefig(out_path, dpi=200)
plt.show()

print(f"[INFO] plot saved to {out_path}")
