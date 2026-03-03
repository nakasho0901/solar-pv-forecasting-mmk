# -*- coding: utf-8 -*-
import os
import sys
import glob
import torch
import numpy as np
import argparse
import importlib.util
from sklearn.metrics import mean_absolute_error, mean_squared_error

# パス設定: solar-kan フォルダを検索パスに追加
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from traingraph import PeakRunner, get_model

# ==========================================================
# 固定定数 (check_scale.py で算出した値)
# ==========================================================
KW_FACTOR = 33.01
OFFSET = 20.54

def find_latest_checkpoint(target_dir):
    """指定されたディレクトリから最新の .ckpt ファイルを探す"""
    ckpt_files = glob.glob(os.path.join(parent_dir, target_dir, "**", "*.ckpt"), recursive=True)
    if not ckpt_files:
        ckpt_files = glob.glob(os.path.join(parent_dir, "lightning_logs", "**", "*.ckpt"), recursive=True)
    
    if not ckpt_files:
        raise FileNotFoundError(f"ディレクトリ '{target_dir}' 内にチェックポイントが見つかりません。")
    
    return max(ckpt_files, key=os.path.getmtime)

def main():
    parser = argparse.ArgumentParser(description="MAE/MSE Score Calculator")
    parser.add_argument("-d", "--dir", type=str, default="save_it_peak")
    parser.add_argument("-c", "--config", type=str, required=True)
    args = parser.parse_args()

    # 1. 設定の読み込み
    spec = importlib.util.spec_from_file_location("config", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    exp_conf = config_module.exp_conf

    # 2. モデルの準備
    try:
        ckpt_path = find_latest_checkpoint(args.dir)
        print(f"評価対象モデル: {ckpt_path}")
    except Exception as e:
        print(f"エラー: {e}")
        return

    # 3. モデルのロード (strict=False 指定)
    base_model = get_model(exp_conf["model_name"], exp_conf)
    try:
        model = PeakRunner.load_from_checkpoint(
            ckpt_path, 
            model=base_model, 
            config=exp_conf,
            strict=False
        )
        model.eval()
    except Exception as e:
        print(f"モデルロード失敗: {e}")
        return

    # 4. テストデータの準備
    from easytsf.runner.data_runner import NPYDataInterface
    data_module = NPYDataInterface(**exp_conf)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    all_preds = []
    all_trues = []

    # 5. 推論実行
    print(f"'{exp_conf['model_name']}' の全テストデータ評価を開始...")
    with torch.no_grad():
        for batch in test_loader:
            var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
            
            # 日射量(Var 2)を1.2倍強調
            var_x_input = var_x.clone()
            var_x_input[:, :, 2] = var_x_input[:, :, 2] * 1.2
            
            # 【修正ポイント】戻り値がいくつあっても、最初の1つ(pred)だけを受け取る
            outputs = model(var_x_input, marker_x)
            if isinstance(outputs, tuple) or isinstance(outputs, list):
                pred = outputs[0]
            else:
                pred = outputs
            
            # 発電量(PV: index 3 or 0)を抽出し、kW単位へ逆正規化
            # モデルによって出力次元が違う場合があるため柔軟に対応
            if pred.shape[-1] >= 4:
                pred_pv = pred[:, :, 3].cpu().numpy()
            else:
                pred_pv = pred[:, :, 0].cpu().numpy()
                
            true_pv = var_y[:, :, 0].cpu().numpy()
            
            pred_kw = (pred_pv * KW_FACTOR) + OFFSET
            true_kw = (true_pv * KW_FACTOR) + OFFSET
            
            all_preds.append(pred_kw.flatten())
            all_trues.append(true_kw.flatten())

    preds = np.concatenate(all_preds)
    trues = np.concatenate(all_trues)

    # 6. 指標計算
    mae = mean_absolute_error(trues, preds)
    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)

    print("\n" + "="*45)
    print(f"  【 {exp_conf['model_name']} 】 評価結果レポート (kW単位)")
    print("="*45)
    print(f"  MAE  (平均絶対誤差)       : {mae:8.4f} kW")
    print(f"  MSE  (平均二乗誤差)       : {mse:8.4f}")
    print(f"  RMSE (二乗平均平方根誤差) : {rmse:8.4f} kW")
    print("-" * 45)
    print(f"  136kWに対する平均誤差率   : {(mae/136.0)*100:6.2f} %")
    print("="*45)

if __name__ == "__main__":
    main()