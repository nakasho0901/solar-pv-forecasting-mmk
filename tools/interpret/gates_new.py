import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import importlib.util

# --- 1. Settings ---
VERSION = "version_12"
PROJECT_ROOT = os.getcwd()
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "tsukuba_conf", "MMK_FusionPV_FeatureToken_A11_stride1_v2.py")
CKPT_PATH = os.path.join(PROJECT_ROOT, "save", "MMK_FusionPV_FeatureToken", VERSION, "checkpoints", "last.ckpt")
DATA_DIR = os.path.join(PROJECT_ROOT, r"dataset_PV\prepared\96-96_fusionA11_96_96_stride1")
TEST_X_PATH = os.path.join(DATA_DIR, "test_x.npy")

# Targeted 6 Features
TARGET_FEATURES = ['pv_kwh', 'ghi_wm2', 'temp_c', 'rh_pct', 'lag_pv_kwh_24h', 'hour_sin']

def load_config(path):
    spec = importlib.util.spec_from_file_location("config_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.exp_conf

def get_model(conf):
    model_file = os.path.join(PROJECT_ROOT, "easytsf", "model", "MMK_FusionPV_FeatureToken_v2.py")
    spec = importlib.util.spec_from_file_location("model_v2", model_file)
    model_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_mod)
    return model_mod.MMK_FusionPV_FeatureToken(
        hist_len=conf["hist_len"], pred_len=conf["pred_len"], var_num=conf["var_num"],
        token_dim=conf.get("token_dim", 64), layer_num=conf.get("layer_num", 3),
        layer_hp=conf["layer_hp"]
    )

def main():
    if not os.path.exists(CKPT_PATH) or not os.path.exists(TEST_X_PATH):
        print("Error: Missing files. Check paths.")
        return

    # Load Config and Data
    conf = load_config(CONFIG_PATH)
    test_x = np.load(TEST_X_PATH)
    conf['var_num'] = test_x.shape[2]

    # Load Model
    model = get_model(conf)
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    sd = {k.replace("model.", ""): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(sd, strict=False)
    model.eval()

    feature_names = ['pv_kwh', 'temp_c', 'rh_pct', 'ghi_wm2', 'hour_sin', 'hour_cos', 'is_daylight', 
                     'lag_pv_kwh_24h', 'lag_ghi_wm2_1h', 'lag_ghi_wm2_2h', 'lag_ghi_wm2_3h', 'lag_ghi_wm2_24h']
    expert_names = ["Spline", "Wav", "Taylor", "Jacobi"]

    # Batch process for ALL test data
    batch_size = 64
    total_samples = len(test_x)
    all_gates_sums = [0, 0, 0]

    print(f"Analyzing FULL test data ({total_samples} samples)...")
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_x = torch.from_numpy(test_x[i:i+batch_size]).float()
            _, gates_batch = model(batch_x, return_gate=True)
            for layer_idx, gate_tensor in enumerate(gates_batch):
                all_gates_sums[layer_idx] += gate_tensor.sum(dim=0).cpu().numpy()

    # --- 保存用のリスト ---
    summary_dataframes = []

    for layer_i, gate_sum in enumerate(all_gates_sums):
        # 数値の平均を算出
        gate_mean = (gate_sum / total_samples).T
        
        # DataFrame作成
        df_full = pd.DataFrame(gate_mean, index=expert_names, columns=feature_names)
        df_selected = df_full[TARGET_FEATURES].copy()
        
        # 1. 各レイヤーの数値をCSVで保存
        csv_name = f"summary_v12_layer{layer_i}_FULL.csv"
        df_selected.to_csv(csv_name)

        # 2. 各レイヤーのヒートマップを保存 (Reds配色)
        plt.figure(figsize=(10, 5))
        sns.heatmap(df_selected, annot=True, fmt=".3f", cmap="Reds", vmin=0, vmax=1, linewidths=0.5, linecolor='gray')
        plt.title(f"Expert Selection (Full Test) - Layer {layer_i}")
        img_name = f"presentation_v12_layer{layer_i}_FULL.png"
        plt.savefig(img_name, dpi=300)
        plt.close()

        # 3. マスターまとめ用のデータ加工
        df_save = df_selected.copy()
        df_save.insert(0, 'Layer', f'Layer {layer_i}')
        summary_dataframes.append(df_save)
        
        print(f"Done Layer {layer_i}: {img_name} & {csv_name}")

    # --- 4. すべてを1つのファイルにまとめる ---
    master_csv = "MASTER_SUMMARY_v12_FULL.csv"
    pd.concat(summary_dataframes).to_csv(master_csv)
    print(f"\n[COMPLETE] All numerical data is bundled in: {master_csv}")

if __name__ == "__main__":
    main()