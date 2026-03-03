# -*- coding: utf-8 -*-
import os, sys, glob, torch
import numpy as np
import pandas as pd
import importlib.util
from sklearn.metrics import mean_absolute_error, mean_squared_error

# パス設定
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from traingraph import PeakRunner, get_model

# 物理量変換定数
KW_FACTOR = 33.01
OFFSET = 20.54

def get_best_checkpoint(model_dir):
    """ディレクトリ内から最も val/loss が低い 'best' チェックポイントを探す"""
    search_pattern = os.path.join(parent_dir, model_dir, "**", "*.ckpt")
    ckpt_files = glob.glob(search_pattern, recursive=True)
    
    # ファイル名から 'best' を含み、かつ val_loss が最小のものを探す
    best_files = [f for f in ckpt_files if "best" in os.path.basename(f)]
    
    if not best_files:
        if not ckpt_files:
            return None
        return max(ckpt_files, key=os.path.getmtime)
    
    # ファイル名から val_loss0.xxxx の数値を抽出して最小のものを選択
    try:
        def extract_loss(path):
            name = os.path.basename(path)
            if "val_loss" in name:
                return float(name.split("val_loss")[-1].replace(".ckpt", "").replace("=", ""))
            return 999.0
        return min(best_files, key=extract_loss)
    except:
        return max(best_files, key=os.path.getmtime)

def evaluate_model_kw(model_dir, config_file):
    ckpt_path = get_best_checkpoint(model_dir)
    if not ckpt_path:
        return None, None, None

    spec = importlib.util.spec_from_file_location("config", os.path.join(parent_dir, config_file))
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    exp_conf = config_module.exp_conf
    
    # データのロード
    from easytsf.runner.data_runner import NPYDataInterface
    data_module = NPYDataInterface(**exp_conf)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    
    # モデルのロード
    base_model = get_model(exp_conf["model_name"], exp_conf)
    model = PeakRunner.load_from_checkpoint(ckpt_path, model=base_model, config=exp_conf, strict=False).eval()
    
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in test_loader:
            var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
            # 日射量強調 (学習時と一致させる)
            var_x_input = var_x.clone(); var_x_input[:, :, 2] *= 1.3
            
            outputs = model(var_x_input, marker_x)
            pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            
            # kW単位に変換 (誤差の計算にオフセットは不要: |(p*S+O)-(t*S+O)| = |p-t|*S)
            p_kw = pred[:, :, 3].cpu().numpy() * KW_FACTOR
            t_kw = var_y[:, :, 0].cpu().numpy() * KW_FACTOR
            
            all_preds.append(p_kw.flatten())
            all_trues.append(t_kw.flatten())

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)
    
    mae = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))
    return mae, rmse, os.path.basename(ckpt_path)

def main():
    print("--- 実測値(kW)ベース 最終精度比較 ---")
    
    # iTransformer 評価
    i_mae, i_rmse, i_file = evaluate_model_kw("save_it_optimized", "config/tsukuba_conf/iTransformer_peak_96.py")
    
    # MMK_Mix 評価
    m_mae, m_rmse, m_file = evaluate_model_kw("save_mmk_optimized", "config/tsukuba_conf/MMK_Mix_PV_96.py")

    results = []
    if i_mae: results.append(["iTransformer", i_mae, i_rmse, i_file])
    if m_mae: results.append(["MMK_Mix", m_mae, m_rmse, m_file])

    df = pd.DataFrame(results, columns=["Model", "MAE (kW)", "RMSE (kW)", "Used Checkpoint"])
    print("\n", df.to_string(index=False))
    
    if len(results) == 2:
        diff = m_mae - i_mae
        print(f"\n[結論] MAEにおいて iTransformer が MMK_Mix より {abs(diff):.4f} kW 精密です。")

if __name__ == "__main__":
    main()