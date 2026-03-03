# -*- coding: utf-8 -*-
import os, sys, glob, torch
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# パス設定
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from traingraph import PeakRunner, get_model

KW_FACTOR = 33.01
OFFSET = 20.54

def get_best_pred(model_dir, config_file):
    # 修正：ファイル名のパターンを広げ、パスの区切り文字が含まれる特殊なファイル名にも対応
    search_pattern = os.path.join(parent_dir, model_dir, "**", "*.ckpt")
    ckpt_files = glob.glob(search_pattern, recursive=True)
    
    # 'last.ckpt'を除外し、'best'を含むファイルを優先的に探す
    best_files = [f for f in ckpt_files if "best" in os.path.basename(f)]
    
    if not best_files:
        if not ckpt_files:
            raise FileNotFoundError(f"ディレクトリ '{model_dir}' 内に .ckpt ファイルが見つかりません。")
        target_file = max(ckpt_files, key=os.path.getmtime)
    else:
        target_file = max(best_files, key=os.path.getmtime)
        
    print(f"ロード中: {os.path.basename(target_file)}")

    spec = importlib.util.spec_from_file_location("config", os.path.join(parent_dir, config_file))
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    exp_conf = config_module.exp_conf
    
    from easytsf.runner.data_runner import NPYDataInterface
    data_module = NPYDataInterface(**exp_conf)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    
    base_model = get_model(exp_conf["model_name"], exp_conf)
    # strict=False でチェックポイント読み込み時の不整合を回避
    model = PeakRunner.load_from_checkpoint(target_file, model=base_model, config=exp_conf, strict=False).eval()
    
    last_batch = None
    for batch in test_loader: 
        last_batch = batch
        
    var_x, marker_x, var_y, marker_y = [_.float() for _ in last_batch]
    
    with torch.no_grad():
        v_x = var_x.clone()
        v_x[:, :, 2] *= 1.3 # 学習時と同じ強調を適用
        outputs = model(v_x, marker_x)
        pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        
        # 最後のサンプルの発電量(index 3)をkW単位に変換
        return (pred[-1, :, 3].cpu().numpy() * KW_FACTOR) + OFFSET, (var_y[-1, :, 0].cpu().numpy() * KW_FACTOR) + OFFSET

def main():
    try:
        print("--- iTransformer の予測をロード中 ---")
        p_it, true = get_best_pred("save_it_optimized", "config/tsukuba_conf/iTransformer_peak_96.py")
        
        print("\n--- MMK_Mix の予測をロード中 ---")
        p_mmk, _ = get_best_pred("save_mmk_optimized", "config/tsukuba_conf/MMK_Mix_PV_96.py")

        plt.figure(figsize=(15, 7))
        plt.plot(true, label="Actual (Measured)", color="black", linestyle="--", alpha=0.6)
        plt.plot(p_it, label="Predicted (iTransformer)", color="#e63946", linewidth=2.5)
        plt.plot(p_mmk, label="Predicted (Optimized MMK_Mix)", color="#2a9d8f", linewidth=2)
        
        plt.title("Final Performance Showdown: iTransformer vs Optimized MMK_Mix", fontsize=14)
        plt.xlabel("Time Steps (96 Hours)")
        plt.ylabel("Power Output [kW]")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(parent_dir, "final_showdown_plot.png")
        plt.savefig(save_path, dpi=300)
        print(f"\n[SUCCESS] 対決グラフを保存しました: {save_path}")
        plt.show()
        
    except Exception as e:
        print(f"\n[ERROR] グラフ作成中にエラーが発生しました: {e}")

if __name__ == "__main__": 
    main()