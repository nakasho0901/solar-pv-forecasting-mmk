# -*- coding: utf-8 -*-
"""
MMK_Mix (24h) の予測結果を kW 単位で評価するスクリプト（mean/std を ckpt 内から総当たりで取得）

実行:
  python tools\\evaluate_mmk_kw_24h.py
"""

import os
import sys
import numpy as np
import torch
import importlib.util

# ===== import パス =====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

CONFIG_PATH = os.path.join(PROJECT_DIR, "config", "tsukuba_conf", "MMK_Mix_PV_24.py")
CKPT_PATH = os.path.join(
    PROJECT_DIR,
    "save", "MMK_Mix_PV_24", "MMK_Mix_PV_24",
    "seed_2025", "version_1", "checkpoints",
    "epoch=epoch=28-step=step=10469.ckpt"
)
DATA_DIR = os.path.join(PROJECT_DIR, "dataset_PV", "processed_24_hardzero")


def load_exp_conf(path: str) -> dict:
    spec = importlib.util.spec_from_file_location("config_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod.exp_conf


def _as_float(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    if isinstance(v, np.ndarray):
        return float(v.reshape(-1)[0])
    return float(v)


def find_mean_std_anywhere(obj, max_depth=6):
    """
    ckpt(dict) のどこかにある mean/std を探す。
    返り値: (found: bool, mean: float, std: float, where: str)
    """
    from collections import deque

    q = deque()
    q.append(("", obj, 0))

    # キー候補（よくある名前揺れも見る）
    mean_keys = {"mean", "y_mean", "target_mean"}
    std_keys = {"std", "y_std", "target_std"}

    while q:
        path, cur, depth = q.popleft()
        if depth > max_depth:
            continue

        if isinstance(cur, dict):
            keys = set(cur.keys())
            if keys & mean_keys and keys & std_keys:
                mk = list(keys & mean_keys)[0]
                sk = list(keys & std_keys)[0]
                try:
                    return True, _as_float(cur[mk]), _as_float(cur[sk]), f"{path}/{mk},{sk}"
                except Exception:
                    pass

            for k, v in cur.items():
                q.append((f"{path}/{k}", v, depth + 1))

        elif isinstance(cur, (list, tuple)):
            for i, v in enumerate(cur):
                q.append((f"{path}[{i}]", v, depth + 1))

    return False, 0.0, 0.0, ""


def main():
    exp_conf = load_exp_conf(CONFIG_PATH)

    # ckpt 読み込み（安全性警告は気にしなくてOK：自分のファイルなら問題なし）
    ckpt = torch.load(CKPT_PATH, map_location="cpu")

    found, mean, std, where = find_mean_std_anywhere(ckpt)

    if found and std != 0.0:
        inv_mode = "z-score(mean/std)"
        print(f"[INFO] Found mean/std in ckpt at: {where}")
    else:
        inv_mode = "scale(max)"
        # fallback（maxで戻す）
        y_train = np.load(os.path.join(DATA_DIR, "train_y.npy"))
        mean = 0.0
        std = float(y_train.max())

    print(f"[INFO] inverse normalization: {inv_mode}")
    print(f"       mean={mean:.6f}, std={std:.6f}")

    # ===== データ読み込み（正規化された y）=====
    y_test = np.load(os.path.join(DATA_DIR, "test_y.npy"))  # (S, 24, 1)

    # ===== モデルロード =====
    from traingraph import PeakRunner, get_model

    model = get_model(exp_conf["model_name"], exp_conf)
    runner = PeakRunner.load_from_checkpoint(
        CKPT_PATH, model=model, config=exp_conf, strict=False
    ).eval()

    device = torch.device("cpu")
    runner.to(device)

    x_test = torch.tensor(np.load(os.path.join(DATA_DIR, "test_x.npy")), dtype=torch.float32)
    marker_test = torch.tensor(np.load(os.path.join(DATA_DIR, "test_marker_y.npy")), dtype=torch.float32)

    # ===== 推論 =====
    with torch.no_grad():
        pred_norm = runner(x_test, marker_test).cpu().numpy()

    # ===== 逆正規化 =====
    if inv_mode.startswith("z-score"):
        pred_kw = pred_norm * std + mean
        true_kw = y_test * std + mean
    else:
        pred_kw = pred_norm * std
        true_kw = y_test * std

    mae = float(np.mean(np.abs(pred_kw - true_kw)))
    mse = float(np.mean((pred_kw - true_kw) ** 2))
    rmse = float(np.sqrt(mse))

    print("=== MMK_Mix 24h : kW-based metrics ===")
    print(f"MAE  (kW)  : {mae:.4f}")
    print(f"RMSE (kW)  : {rmse:.4f}")
    print(f"MSE  (kW^2): {mse:.4f}")


if __name__ == "__main__":
    main()
