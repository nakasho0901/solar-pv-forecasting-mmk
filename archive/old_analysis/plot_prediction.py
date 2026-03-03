# -*- coding: utf-8 -*-
import os
import sys
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import importlib.util

# ==========================================================
# 検索パスの追加: 
# tools フォルダから実行しても、一階層上の root (solar-kan) を探せるようにします
# ==========================================================
current_script_dir = os.path.dirname(os.path.abspath(__file__)) # tools フォルダ
parent_dir = os.path.dirname(current_script_dir)               # solar-kan フォルダ
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# これで一階層上の traingraph.py が読み込めるようになります
try:
    from traingraph import PeakRunner, get_model
except ImportError:
    print("エラー: traingraph.py が見つかりません。")
    print(f"現在の検索パス: {sys.path[0]}")
    sys.exit(1)

# ==========================================================
# 設定
# ==========================================================
CONFIG_PATH = os.path.join(parent_dir, "config", "tsukuba_conf", "iTransformer_peak_96.py")

def find_latest_checkpoint():
    """最新のチェックポイントファイルを自動で探す"""
    # 検索対象のフォルダ候補
    search_paths = [
        os.path.join(parent_dir, "save_it_peak"),
        os.path.join(parent_dir, "lightning_logs"),
        "save_it_peak",
        "lightning_logs"
    ]
    
    ckpt_files = []
    for path in search_paths:
        ckpt_files.extend(glob.glob(os.path.join(path, "**", "*.ckpt"), recursive=True))
    
    if not ckpt_files:
        raise FileNotFoundError("チェックポイントファイル（.ckpt）が見つかりません。")

    # 更新日時が一番新しいファイルを選択
    latest_ckpt = max(ckpt_files, key=os.path.getmtime)
    return latest_ckpt

def main():
    # 1. 最新のファイルを自動で見つける
    try:
        checkpoint_path = find_latest_checkpoint()
        print(f"使用する学習済みモデル: {checkpoint_path}")
    except Exception as e:
        print(f"エラー: {e}")
        return

    # 2. Configの読み込み
    spec = importlib.util.spec_from_file_location("config", CONFIG_PATH)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    exp_conf = config_module.exp_conf

    # 3. データの準備
    from easytsf.runner.data_runner import NPYDataInterface
    data_module = NPYDataInterface(**exp_conf)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    # 4. モデルのロード
    base_model = get_model(exp_conf["model_name"], exp_conf)
    model = PeakRunner.load_from_checkpoint(checkpoint_path, model=base_model)
    model.eval()
    model.freeze()

    # 5. 推論 (テストデータの最初のバッチ)
    batch = next(iter(test_loader))
    var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
    
    with torch.no_grad():
        # 日射量強調
        var_x_input = var_x.clone()
        var_x_input[:, :, 2] = var_x_input[:, :, 2] * 1.2
        pred, _, _ = model(var_x_input, marker_x)
        
        # PV(発電量)はインデックス3
        pred_pv = pred[:, :, 3].numpy() 
        true_pv = var_y[:, :, 0].numpy()

    # 6. 最も高いピークがあるサンプルを探してプロット
    sample_idx = np.argmax(np.max(true_pv, axis=1))
    
    plt.figure(figsize=(12, 6))
    plt.plot(true_pv[sample_idx], label="Actual (Measured)", color="black", linestyle="--", alpha=0.6)
    plt.plot(pred_pv[sample_idx], label="Predicted (iTransformer Peak)", color="red", linewidth=2.5)
    
    plt.title(f"Comparison: Actual vs Predicted Peak (Sample {sample_idx})")
    plt.xlabel("Time Steps (15-min intervals over 24h x 4days)")
    plt.ylabel("Normalized Power Output")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # グラフの保存
    save_img = os.path.join(parent_dir, "peak_comparison_graph.png")
    plt.savefig(save_img)
    print(f"グラフを保存しました: {save_img}")
    plt.show()

if __name__ == "__main__":
    main()