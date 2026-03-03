# -*- coding: utf-8 -*-
import os, sys, glob, torch
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# パス設定
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_script_dir)
sys.path.insert(0, parent_dir)
from traingraph import PeakRunner, get_model

# 定数
KW_FACTOR = 33.01
OFFSET = 20.54
TARGET_PEAK = 136.0

def get_prediction(dir_name, config_path):
    ckpt_path = max(glob.glob(os.path.join(parent_dir, dir_name, "**", "*.ckpt"), recursive=True), key=os.path.getmtime)
    spec = importlib.util.spec_from_file_location("config", os.path.join(parent_dir, config_path))
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    exp_conf = config_module.exp_conf

    from easytsf.runner.data_runner import NPYDataInterface
    data_module = NPYDataInterface(**exp_conf)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    
    base_model = get_model(exp_conf["model_name"], exp_conf)
    # strict=False で読み込み
    model = PeakRunner.load_from_checkpoint(ckpt_path, model=base_model, config=exp_conf, strict=False).eval()

    last_batch = None
    for batch in test_loader: last_batch = batch
    var_x, marker_x, var_y, marker_y = [_.float() for _ in last_batch]
    
    with torch.no_grad():
        # 日射量強調
        v_x = var_x.clone()
        v_x[:, :, 2] = v_x[:, :, 2] * 1.2
        outputs = model(v_x, marker_x)
        pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        
        # 最後のサンプル(index -1)を抽出
        p_pv = pred[-1, :, 3].numpy() if pred.shape[-1] >= 4 else pred[-1, :, 0].numpy()
        t_pv = var_y[-1, :, 0].numpy()
    
    return (t_pv * KW_FACTOR) + OFFSET, (p_pv * KW_FACTOR) + OFFSET, exp_conf["model_name"]

def main():
    print("iTransformer の予測を取得中...")
    true, pred_it, name_it = get_prediction("save_it_peak", "config/tsukuba_conf/iTransformer_peak_96.py")
    print("MMK_Mix の予測を取得中...")
    _, pred_mmk, name_mmk = get_prediction("save_mix", "config/tsukuba_conf/MMK_Mix_PV_96.py")

    plt.figure(figsize=(15, 8))
    plt.plot(true, label="Actual (Measured)", color="black", linestyle="--", alpha=0.5)
    plt.plot(pred_it, label=f"Predicted ({name_it})", color="#e63946", linewidth=2.5)
    plt.plot(pred_mmk, label=f"Predicted ({name_mmk})", color="#2a9d8f", linewidth=2)
    
    plt.axhline(y=TARGET_PEAK, color='#1d3557', linestyle=':', label=f"Target Peak ({TARGET_PEAK}kW)")
    plt.title(f"Final Comparison: iTransformer (Error {7.25}%) vs MMK_Mix (Error {22.35}%)", fontsize=14)
    plt.ylabel("Power Output [kW]")
    plt.ylim(0, 160)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    main()