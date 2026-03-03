# -*- coding: utf-8 -*-
import os, sys, torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import importlib.util

# パス設定
CKPT_PATH = r"C:\Users\nakas\my_project\solar-kan\save\MMK_Mix_PV_96\MMK_Mix_PV_96\seed_2025\version_1\checkpoints\epoch=epoch=1-step=step=722.ckpt"
CONFIG_PATH = r"C:\Users\nakas\my_project\solar-kan\config\tsukuba_conf\MMK_Mix_PV_96.py"

sys.path.append(r"C:\Users\nakas\my_project\solar-kan")
from traingraph import PeakRunner, get_model
from easytsf.runner.data_runner import NPYDataInterface

def create_var_expert_heatmap():
    spec = importlib.util.spec_from_file_location("config", CONFIG_PATH)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    exp_conf = config_module.exp_conf
    exp_conf["hidden_dim"] = 96 

    data_module = NPYDataInterface(**exp_conf)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    base_model = get_model(exp_conf["model_name"], exp_conf)
    model = PeakRunner.load_from_checkpoint(CKPT_PATH, model=base_model, config=exp_conf, strict=False).eval()
    if torch.cuda.is_available(): model = model.cuda()

    # 4変数：発電量, 気温, 日射量, 湿度 [cite: 345-349]
    var_names = ["Power", "Temp", "Solar", "Humidity"]
    # 4エキスパート：KAN, WavKAN, TaylorKAN, JacobiKAN [cite: 663]
    expert_names = ["KAN", "WavKAN", "TaylorKAN", "JacobiKAN"]
    
    num_vars = len(var_names)
    num_experts = len(expert_names)
    selection_counts = np.zeros((num_vars, num_experts))

    captured_weights = []
    def hook_fn(module, input, output):
        # output[1] がゲーティング重み。これを確実に取得
        captured_weights.append(output[1].detach().cpu())

    # 最後のMoK層にフックを登録
    handle = model.model.layers[-1].mok.register_forward_hook(hook_fn)

    print("【開始】テストデータのゲーティング統計を集計中...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            var_x, marker_x, _, _ = [b.cuda() if torch.cuda.is_available() else b for b in batch]
            _ = model(var_x, marker_x)
            
            if not captured_weights: continue
            weights = captured_weights.pop(0) # (Batch, num_vars, num_experts)
            
            # 各サンプル・各変数で最大の重みを持つExpertを取得
            top1_ids = torch.argmax(weights, dim=-1) 

            # --- 修正箇所：次元を (Batch, num_vars) に固定する ---
            if top1_ids.ndim == 1:
                top1_ids = top1_ids.unsqueeze(0)
            
            # カウント集計
            for v in range(num_vars):
                # top1_ids[:, v] が (Batch,) の形になるように。
                for e in range(num_experts):
                    selection_counts[v, e] += (top1_ids[:, v] == e).sum().item()

    handle.remove()

    # 正規化（各変数の合計を1.0にする）[cite: 465]
    row_sums = selection_counts.sum(axis=1, keepdims=True)
    heatmap_data = np.divide(selection_counts, row_sums, out=np.zeros_like(selection_counts), where=row_sums!=0)

    # ヒートマップ描画
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Reds", 
                xticklabels=expert_names, yticklabels=var_names)
    plt.title("MMK Variable-Expert Assignment (Research Style)")
    plt.xlabel("Expert KAN Type")
    plt.ylabel("Input Variables")
    
    plt.savefig("research_style_heatmap.png", dpi=300, bbox_inches='tight')
    print("【完了】図を保存しました: research_style_heatmap.png")
    plt.show()

if __name__ == "__main__":
    create_var_expert_heatmap()