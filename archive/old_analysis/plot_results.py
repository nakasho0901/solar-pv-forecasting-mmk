import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import importlib.util
import glob

# プロジェクトルートをパスに追加
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from easytsf.runner.data_runner import NPYDataInterface as DataInterface
from easytsf.runner.exp_runner import LTSFRunner

def load_config(config_path):
    module_name = os.path.splitext(os.path.basename(config_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.exp_conf

def main():
    config_path = os.path.join(root_dir, "config", "tsukuba_conf", "MMK_Mix_PV_96.py")
    conf = load_config(config_path)
    
    ckpt_files = glob.glob(os.path.join(root_dir, "save_mix", "**", "*.ckpt"), recursive=True)
    if not ckpt_files: return
    full_ckpt_path = sorted(ckpt_files, key=os.path.getmtime)[-1]
    
    model = LTSFRunner.load_from_checkpoint(full_ckpt_path, **conf)
    model.eval().cpu()
    
    data_module = DataInterface(**conf)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    
    last_batch = None
    for batch in test_loader:
        last_batch = batch
    
    var_x, marker_x, var_y, marker_y = [_.float() for _ in last_batch]
    
    with torch.no_grad():
        pred_output, _, _ = model.forward(last_batch, batch_idx=0)
        
        true_norm = var_y[-1, :, 0].numpy()
        pred_norm = pred_output[-1, :, 0].numpy()
        
        calib_std = 300.0
        calib_mean = -np.min(true_norm) * calib_std 
        
        true_kw = true_norm * calib_std + calib_mean
        pred_kw = pred_norm * calib_std + calib_mean
        
        pred_kw[true_kw < 5] = 0
        true_kw[true_kw < 5] = 0

    plt.figure(figsize=(15, 6))
    plt.plot(true_kw, label="Actual Power [kW]", color="royalblue", linewidth=2)
    plt.plot(pred_kw, label="MMK_Mix Prediction [kW]", color="crimson", linestyle="--", linewidth=2)
    
    plt.title("Tsukuba PV Generation - Final 96 Hours (kW Scale)", fontsize=16)
    plt.ylabel("Power Output [kW]", fontsize=14)
    plt.xlabel("Time [Hours]", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- 【修正点】縦軸の範囲を拡大 ---
    plt.ylim(-10, 1200) # 1000から1200へ変更

    output_path = os.path.join(root_dir, "pv_96h_final_fixed.png")
    plt.savefig(output_path)
    print(f"[SUCCESS] {output_path} を保存しました。")

if __name__ == "__main__":
    main()