# -*- coding: utf-8 -*-
import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# 1. パス設定（一階層上の traingraph.py を読み込めるようにする）
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from traingraph import PeakRunner, get_model

# 2. 算出した定数と設定
KW_FACTOR = 33.01
OFFSET = 20.54
TARGET_PEAK = 136.0
CONFIG_PATH = os.path.join(parent_dir, "config", "tsukuba_conf", "iTransformer_peak_96.py")

def find_latest_checkpoint():
    """最新のチェックポイントを自動検索"""
    search_paths = [os.path.join(parent_dir, "save_it_peak"), "save_it_peak", os.path.join(parent_dir, "lightning_logs")]
    ckpt_files = []
    for path in search_paths:
        ckpt_files.extend(glob.glob(os.path.join(path, "**", "*.ckpt"), recursive=True))
    if not ckpt_files:
        raise FileNotFoundError("学習済みモデル（.ckpt）が見つかりません。学習が完了しているか確認してください。")
    return max(ckpt_files, key=os.path.getmtime)

def main():
    # モデルのロード
    ckpt_path = find_latest_checkpoint()
    print(f"使用モデル: {ckpt_path}")

    spec = importlib.util.spec_from_file_location("config", CONFIG_PATH)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    exp_conf = config_module.exp_conf

    # データのロード
    from easytsf.runner.data_runner import NPYDataInterface
    data_module = NPYDataInterface(**exp_conf)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    # モデルの構築
    base_model = get_model(exp_conf["model_name"], exp_conf)
    model = PeakRunner.load_from_checkpoint(ckpt_path, model=base_model)
    model.eval()

    # ------------------------------------------------------
    # 【重要】テストデータの「本当の最後」のバッチまで移動
    # ------------------------------------------------------
    print("テストデータの最後まで移動中...")
    last_batch = None
    for batch in test_loader:
        last_batch = batch
    
    var_x, marker_x, var_y, marker_y = [_.float() for _ in last_batch]
    
    with torch.no_grad():
        var_x_input = var_x.clone()
        var_x_input[:, :, 2] = var_x_input[:, :, 2] * 1.2 # 日射量強調
        pred, _, _ = model(var_x_input, marker_x)
        
        # kW単位に変換 (逆正規化)
        pred_kw = (pred[:, :, 3].numpy() * KW_FACTOR) + OFFSET
        true_kw = (var_y[:, :, 0].numpy() * KW_FACTOR) + OFFSET

    # ------------------------------------------------------
    # バッチ内の「一番最後」のサンプルを選択
    # ------------------------------------------------------
    sample_idx = -1 
    
    # グラフ作成
    plt.figure(figsize=(15, 7))
    plt.plot(true_kw[sample_idx], label="Actual (Measured)", color="#333333", linestyle="--", alpha=0.8)
    plt.plot(pred_kw[sample_idx], label="Predicted (iTransformer Peak)", color="#e63946", linewidth=2.5)
    
    # ターゲットライン
    plt.axhline(y=TARGET_PEAK, color='#1d3557', linestyle=':', label=f"Target Peak ({TARGET_PEAK}kW)")
    
    plt.title(f"Final Test Window Analysis: iTransformer vs {TARGET_PEAK}kW", fontsize=14)
    plt.xlabel("Time Steps (Relative to the end of data)", fontsize=12)
    plt.ylabel("Power Output [kW]", fontsize=12)
    plt.ylim(0, 160)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    save_img = os.path.join(parent_dir, "final_peak_comparison.png")
    plt.savefig(save_img, dpi=300)
    print(f"グラフを保存しました: {save_img}")
    plt.show()

if __name__ == "__main__":
    main()