# -*- coding: utf-8 -*-
r"""
MMK FusionPV Gate Analysis (Latest Checkpoint)
----------------------------------------------------
Target: Visualize Expert Selection Gates using the latest retrained model.
"""

import os
import sys
import json
import importlib.util
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch

# ---------------------------------------------------------
# 1. パス設定
# ---------------------------------------------------------
PROJECT_ROOT = os.getcwd()

# 設定ファイル（A9ファイル名でも中身でvar_numを自動補正します）
CONFIG_PATH = os.path.join(PROJECT_ROOT, r"C:\Users\nakas\my_project\solar-kan\config\tsukuba_conf\MMK_FusionPV_FeatureToken_A11_stride1_v2.py")

# ★ここが重要：最新のチェックポイントを指定
# 通常は version_1 の last.ckpt が最新ですが、フォルダが変わった場合は修正してください
CKPT_PATH = os.path.join(PROJECT_ROOT, r"save\MMK_FusionPV_FeatureToken\version_12\checkpoints\last.ckpt")

# データセット (A11)
DATA_DIR = os.path.join(PROJECT_ROOT, r"dataset_PV\prepared\96-96_fusionA11_96_96_stride1")
META_JSON = os.path.join(DATA_DIR, "meta.json")

# 出力先
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "research_gate_results_latest")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# 2. ユーティリティ
# ---------------------------------------------------------
def load_config_from_file(path):
    spec = importlib.util.spec_from_file_location("config_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.exp_conf

def load_feature_names(meta_path, default_num=12):
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            names = data.get("columns", {}).get("feature_cols", [])
            if len(names) > 0:
                print(f"Feature Names Loaded: {names}")
                return names
    print("Meta json not found or empty. Using default names.")
    return [f"Feat_{i}" for i in range(default_num)]

def import_model_class():
    sys.path.append(PROJECT_ROOT)
    # 階層構造が変わっている場合に対応
    try:
        from easytsf.model.MMK_FusionPV_FeatureToken import MMK_FusionPV_FeatureToken
        return MMK_FusionPV_FeatureToken
    except ImportError:
        # 直接パスからロード
        model_file = os.path.join(PROJECT_ROOT, "MMK_FusionPV_FeatureToken.py")
        if not os.path.exists(model_file):
             # easytsfフォルダ内を探す
             model_file = os.path.join(PROJECT_ROOT, "easytsf", "model", "MMK_FusionPV_FeatureToken.py")
        
        spec = importlib.util.spec_from_file_location("MMK_FusionPV_FeatureToken", model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.MMK_FusionPV_FeatureToken

# ---------------------------------------------------------
# 3. メイン処理
# ---------------------------------------------------------
def main():
    print(f"--- Gate Analysis (Latest) Started on {DEVICE} ---")

    # Load Data first to determine input dimension
    test_x_path = os.path.join(DATA_DIR, "test_x.npy")
    if not os.path.exists(test_x_path):
        # カレントディレクトリも探す
        if os.path.exists("test_x.npy"):
            test_x_path = "test_x.npy"
            DATA_DIR_LOCAL = "."
        else:
            print("【エラー】test_x.npy が見つかりません。")
            return
    else:
        DATA_DIR_LOCAL = DATA_DIR

    data_np = np.load(test_x_path)
    real_var_num = data_np.shape[2]
    print(f"Data Loaded: shape={data_np.shape}, var_num={real_var_num}")

    # Load Config
    if not os.path.exists(CONFIG_PATH):
        # カレントディレクトリを探す
        if os.path.exists("MMK_FusionPV_Flatten_A9_96.py"):
             CONFIG_PATH_LOCAL = "MMK_FusionPV_Flatten_A9_96.py"
        else:
             print("【エラー】Configファイルが見つかりません。")
             return
        conf = load_config_from_file(CONFIG_PATH_LOCAL)
    else:
        conf = load_config_from_file(CONFIG_PATH)

    # Override var_num
    conf['var_num'] = real_var_num
    feature_names = load_feature_names(META_JSON, default_num=real_var_num)

    # Build Model
    print("Building Model...")
    ModelClass = import_model_class()
    model = ModelClass(
        hist_len=conf["hist_len"], pred_len=conf["pred_len"], var_num=conf["var_num"],
        pv_index=conf.get("pv_index", 0), token_dim=conf.get("token_dim", 64),
        layer_num=conf.get("layer_num", 3), layer_hp=conf.get("layer_hp"),
        use_layernorm=conf.get("use_layernorm", True), dropout=conf.get("dropout", 0.1),
        fusion=conf.get("fusion", "mean"), baseline_mode=conf.get("baseline_mode", "last24_repeat"),
        use_pv_revin=conf.get("use_pv_revin", True), revin_eps=conf.get("revin_eps", 1e-5),
        enforce_nonneg=conf.get("enforce_nonneg", True),
    )

    # Load Weights
    print(f"Loading Checkpoint: {CKPT_PATH}")
    if not os.path.exists(CKPT_PATH):
        # カレントディレクトリのlast.ckptを探す（アップロードされた場合）
        if os.path.exists("last.ckpt"):
            CKPT_PATH_LOCAL = "last.ckpt"
            print("Using 'last.ckpt' in current directory.")
        else:
            print(f"【エラー】チェックポイントが見つかりません: {CKPT_PATH}")
            return
    else:
        CKPT_PATH_LOCAL = CKPT_PATH

    ckpt = torch.load(CKPT_PATH_LOCAL, map_location=DEVICE)
    state_dict = {k.replace("model.", ""): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()

    # Inference
    print("Extracting Gates...")
    # Use first batch
    batch_size = 128
    input_tensor = torch.from_numpy(data_np[:batch_size]).float().to(DEVICE)

    with torch.no_grad():
        _, gates_all = model(input_tensor, return_gate=True)

    # Visualization
    print("Plotting Heatmaps...")
    expert_names = ["KAN (Spline)", "WavKAN", "TaylorKAN", "JacobiKAN"]
    plt.rcParams['font.family'] = 'sans-serif'

    for layer_i, gate_tensor in enumerate(gates_all):
        if layer_i >= 3: break 
        
        # gate_tensor: (B, F, E) -> mean over batch -> (F, E)
        gate_mean = gate_tensor.mean(dim=0).cpu().numpy() 
        heatmap_data = gate_mean.T # (Experts, Features)

        # Ensure columns match
        current_cols = feature_names if len(feature_names) == heatmap_data.shape[1] else [f"F{i}" for i in range(heatmap_data.shape[1])]

        df = pd.DataFrame(heatmap_data, index=expert_names, columns=current_cols)
        
        # Save CSV
        df.to_csv(os.path.join(OUTPUT_DIR, f"layer{layer_i}_gate_matrix.csv"))

        # Plot
        plt.figure(figsize=(16, 7))
        sns.set_theme(style="white")
        ax = sns.heatmap(
            df, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1,
            linewidths=1.0, linecolor='white',
            cbar_kws={'label': 'Selection Probability'}
        )
        plt.title(f"MMK Expert Strategy (Layer {layer_i}) - Latest", fontsize=18, fontweight='bold')
        plt.xlabel("Features", fontsize=14)
        plt.ylabel("Experts", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.yticks(rotation=0, fontsize=12)
        plt.tight_layout()

        out_path = os.path.join(OUTPUT_DIR, f"heatmap_layer{layer_i}_latest.png")
        plt.savefig(out_path, dpi=300)
        print(f"  Saved: {out_path}")

    print(f"\n【完了】解析結果は '{OUTPUT_DIR}' に保存されました。")

if __name__ == "__main__":
    main()