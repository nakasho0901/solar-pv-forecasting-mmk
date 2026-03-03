# -*- coding: utf-8 -*-
import os, sys, glob, torch
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import matplotlib.gridspec as gridspec

# 1. 環境設定
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path: sys.path.insert(0, current_dir)
from traingraph import PeakRunner, get_model
from easytsf.runner.data_runner import NPYDataInterface

def main():
    # --- 修正点：ファイルを save_mix 内から自動で探す ---
    search_pattern = os.path.join(current_dir, "save_mix", "**", "*loss=0.4605.ckpt")
    found_files = glob.glob(search_pattern, recursive=True)
    
    if not found_files:
        print("【エラー】loss=0.4605.ckpt が見つかりません。パスを確認してください。")
        return
    ckpt_path = found_files[0]
    print(f"【ロード成功】{ckpt_path}")

    config_file = "config/tsukuba_conf/MMK_Mix_PV_96.py"
    spec = importlib.util.spec_from_file_location("config", os.path.join(current_dir, config_file))
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    exp_conf = config_module.exp_conf
    exp_conf["hidden_dim"] = 96

    # 2. データのロード
    data_module = NPYDataInterface(**exp_conf)
    data_module.setup("test")
    test_dataset = data_module.test_dataloader().dataset
    batch = test_dataset[-1] 
    var_x, marker_x, var_y, marker_y = [torch.as_tensor(_).unsqueeze(0).float() for _ in batch]

    # 3. モデル構築
    base_model = get_model(exp_conf["model_name"], exp_conf)
    model = PeakRunner.load_from_checkpoint(ckpt_path, model=base_model, config=exp_conf, strict=False).eval()
    
    captured_scores = []
    def hook_fn(module, input, output):
        captured_scores.append(output[1].detach().cpu())
    model.model.layers[-1].mok.register_forward_hook(hook_fn)

    with torch.no_grad():
        v_x = var_x.clone(); v_x[:, :, 2] *= 1.3
        outputs = model(v_x, marker_x)
        pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        
        preds_kw = pred[0, :, 3].cpu().numpy() * 33.01 + 20.54
        trues_kw = var_y[0, :, 0].cpu().numpy() * 33.01 + 20.54

        # 4. 重要：データのインデックスと名前を一致させる
        # hparams.yaml の順序: 0:KAN, 1:WavKAN, 2:TaylorKAN, 3:JacobiKAN
        scores = captured_scores[0].view(1, 8, -1)[0, 3, :].numpy()
        expert_names = ["KAN", "WavKAN", "TaylorKAN", "JacobiKAN"]
        
        # imshowで上から順に表示するために整形
        gating_heatmap = np.tile(scores[:, np.newaxis], (1, 96))

    # 5. 描画
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.15)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1], sharex=ax1)

    ax1.plot(trues_kw, 'k--', label='Actual', alpha=0.5)
    ax1.plot(preds_kw, 'r', label='MMK (96-dim)')
    ax1.set_ylabel("Power [kW]")
    ax1.legend()
    ax1.set_title(f"MMK MoK Analysis (Corrected Labels)")

    # 修正：データの並び(0=KAN)とラベルを一致させて描画
    im = ax2.imshow(gating_heatmap, aspect='auto', cmap='magma', 
                    extent=[-0.5, 95.5, len(expert_names)-0.5, -0.5], vmin=0, vmax=1.0)
    
    ax2.set_yticks(np.arange(len(expert_names)))
    ax2.set_yticklabels(expert_names)
    ax2.set_ylabel("Experts")
    
    plt.colorbar(im, ax=ax2, label='Selection Weight')
    plt.savefig("final_confirmed_heatmap.png", dpi=300)
    print("【完了】ラベルが保証された図を保存しました: final_confirmed_heatmap.png")
    plt.show()

if __name__ == "__main__":
    main()