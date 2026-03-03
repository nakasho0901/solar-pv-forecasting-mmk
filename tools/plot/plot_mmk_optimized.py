# -*- coding: utf-8 -*-
import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# 1. パス設定
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from traingraph import PeakRunner, get_model

# 2. 定数（kW変換用）
KW_FACTOR = 33.01
OFFSET = 20.54
TARGET_PEAK = 136.0
# 最適化学習で使ったConfigパス
CONFIG_PATH = os.path.join(parent_dir, "config", "tsukuba_conf", "MMK_Mix_PV_96.py")
# 学習結果の保存ディレクトリ
SAVE_DIR = os.path.join(parent_dir, "save_mmk_optimized")

def find_best_checkpoint():
    """保存されたチェックポイントの中から、最も val/loss が低い『最良』のファイルを探す"""
    # MMK_Mix フォルダ内のすべての .ckpt を検索
    ckpt_files = glob.glob(os.path.join(SAVE_DIR, "**", "best-*.ckpt"), recursive=True)
    if not ckpt_files:
        # best がなければ last を探す
        ckpt_files = glob.glob(os.path.join(SAVE_DIR, "**", "last.ckpt"), recursive=True)
    
    if not ckpt_files:
        raise FileNotFoundError(f"ディレクトリ '{SAVE_DIR}' 内にチェックポイントが見つかりません。学習が完了しているか確認してください。")
    
    # ファイル名から val/loss を抽出して最小のものを返す、または最新のものを返す
    # ここではシンプルに更新日時が最も新しいものを選択します
    return max(ckpt_files, key=os.path.getmtime)

def main():
    # モデルのロード
    ckpt_path = find_best_checkpoint()
    print(f"使用するベストモデル: {ckpt_path}")

    spec = importlib.util.spec_from_file_location("config", CONFIG_PATH)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    exp_conf = config_module.exp_conf

    # データのロード
    from easytsf.runner.data_runner import NPYDataInterface
    data_module = NPYDataInterface(**exp_conf)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    # モデルの構築 (strict=False で統計量の不一致を回避)
    base_model = get_model(exp_conf["model_name"], exp_conf)
    model = PeakRunner.load_from_checkpoint(ckpt_path, model=base_model, config=exp_conf, strict=False)
    model.eval()

    # 最後のバッチを取得
    last_batch = None
    for batch in test_loader:
        last_batch = batch
    
    var_x, marker_x, var_y, marker_y = [_.float() for _ in last_batch]
    
    with torch.no_grad():
        # 学習時と同じく日射量を強調
        var_x_input = var_x.clone()
        var_x_input[:, :, 2] = var_x_input[:, :, 2] * 1.3 
        
        outputs = model(var_x_input, marker_x)
        
        # MMK_Mix の戻り値形式に対応
        if isinstance(outputs, (tuple, list)):
            pred = outputs[0]
        else:
            pred = outputs
            
        # 発電量(PV: index 3)を抽出
        # pred は [Batch, Pred_len, Variable]
        pred_pv = pred[:, :, 3].numpy()
        true_pv = var_y[:, :, 0].numpy() # label は [Batch, Pred_len, 1]

        # kW単位に変換 (逆正規化)
        pred_kw = (pred_pv * KW_FACTOR) + OFFSET
        true_kw = (true_pv * KW_FACTOR) + OFFSET

    # グラフ作成 (最後のサンプルを表示)
    sample_idx = -1 
    
    plt.figure(figsize=(15, 7))
    plt.plot(true_kw[sample_idx], label="Actual (Measured)", color="#333333", linestyle="--", alpha=0.8)
    plt.plot(pred_kw[sample_idx], label="Predicted (Optimized MMK_Mix)", color="#2a9d8f", linewidth=2.5)
    
    # ターゲットライン
    plt.axhline(y=TARGET_PEAK, color='#1d3557', linestyle=':', label=f"Target Peak ({TARGET_PEAK}kW)")
    
    plt.title(f"Optimized MMK_Mix Analysis: Final 96 Hours vs {TARGET_PEAK}kW", fontsize=14)
    plt.xlabel("Time Steps (Hours)", fontsize=12)
    plt.ylabel("Power Output [kW]", fontsize=12)
    plt.ylim(0, 160)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    save_img = os.path.join(parent_dir, "optimized_mmk_prediction.png")
    plt.savefig(save_img, dpi=300)
    print(f"グラフを保存しました: {save_img}")
    plt.show()

if __name__ == "__main__":
    main()