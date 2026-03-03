# -*- coding: utf-8 -*-
"""
MMK TaylorKAN Formula Extractor (Version 1 / A11 Dataset)
---------------------------------------------------------
Target: Extract Taylor polynomials from the new model (11/12 features).
Features: Automatically detects var_num from meta.json to match checkpoint.
"""

import os
import sys
import json
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import torch

# ---------------------------------------------------------
# 1. User Environment Settings (Updated for Version 1)
# ---------------------------------------------------------
PROJECT_ROOT = os.getcwd()

# Config path (Even if name is A9, we will override var_num from dataset)
CONFIG_PATH = os.path.join(PROJECT_ROOT, r"config\tsukuba_conf\MMK_FusionPV_Flatten_A9_96.py")

# New Checkpoint path (Version 1)
CKPT_PATH = os.path.join(PROJECT_ROOT, r"save\MMK_FusionPV_FeatureToken\version_1\checkpoints\last.ckpt")

# New Dataset path (A11)
DATA_DIR = os.path.join(PROJECT_ROOT, r"dataset_PV\prepared\96-96_fusionA11_96_96_stride1")
META_JSON = os.path.join(DATA_DIR, "meta.json")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "research_formula_analysis_v1")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------
def load_config_from_file(path):
    spec = importlib.util.spec_from_file_location("config_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.exp_conf

def get_var_num_from_meta(meta_path):
    """Detect correct number of variables from meta.json"""
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            cols = data.get("columns", {}).get("feature_cols", [])
            print(f"【自動検出】変数数: {len(cols)} (from meta.json)")
            return len(cols)
    else:
        print("【警告】meta.jsonが見つかりません。Configの値をそのまま使います。")
        return None

def import_model_class():
    sys.path.append(PROJECT_ROOT)
    try:
        from easytsf.model.MMK_FusionPV_FeatureToken import MMK_FusionPV_FeatureToken
        return MMK_FusionPV_FeatureToken
    except ImportError:
        model_file = os.path.join(PROJECT_ROOT, "easytsf", "model", "MMK_FusionPV_FeatureToken.py")
        if not os.path.exists(model_file):
             print(f"Model file not found: {model_file}")
             sys.exit(1)
        spec = importlib.util.spec_from_file_location("MMK_FusionPV_FeatureToken", model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.MMK_FusionPV_FeatureToken

# ---------------------------------------------------------
# 3. Main Analysis Logic
# ---------------------------------------------------------
def analyze_taylor_layer_force(model, layer_idx, expert_idx=2):
    """Extract coefficients from TaylorKAN (Expert Index 2)."""
    print(f"\n--- Analyzing Layer {layer_idx}, Expert {expert_idx} (TaylorKAN) ---")
    
    if layer_idx >= len(model.blocks):
        print(f"Error: Layer {layer_idx} does not exist.")
        return None

    block = model.blocks[layer_idx]
    
    if not hasattr(block, 'mok'):
        print("Block does not have 'mok' attribute.")
        return None
    
    experts = block.mok.experts
    if expert_idx >= len(experts):
         print(f"Expert index {expert_idx} out of range.")
         return None
         
    target_expert = experts[expert_idx]
    coeffs = None
    
    print("Searching for coefficients...")
    for name, param in target_expert.named_parameters():
        shape = tuple(param.shape)
        # Look for shape ending in 3~10 (Polynomial coeffs)
        if len(shape) >= 2 and (3 <= shape[-1] <= 10):
            coeffs = param.detach().cpu().numpy()
            print(f"  -> HIT! Found polynomial coefficients: {name} shape={shape}")
            break
            
    if coeffs is None:
        print("Could not identify coefficient parameters.")
        return None

    # Calculate Average Polynomial
    if coeffs.ndim == 3:
        mean_coeffs = np.mean(coeffs, axis=(0, 1))
    elif coeffs.ndim == 2:
        mean_coeffs = np.mean(coeffs, axis=0)
    else:
        mean_coeffs = coeffs.flatten()

    # Format Formula string
    formula = "y = "
    for i, c in enumerate(mean_coeffs):
        sign = "+" if c >= 0 else "-"
        formula += f"{sign} {abs(c):.5f}x^{i} "
    
    print(f"Formula: {formula}")
    return mean_coeffs

def plot_polynomials(layer_coeffs, output_path):
    x = np.linspace(-1.5, 1.5, 100)
    
    plt.figure(figsize=(10, 6))
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)

    colors = {0: 'blue', 1: 'green', 2: 'red'}
    labels = {0: 'Layer 0 (Input)', 1: 'Layer 1', 2: 'Layer 2 (Output/Deep)'}

    for layer_idx, coeffs in layer_coeffs.items():
        if coeffs is None: continue
        
        y = np.zeros_like(x)
        for power, c in enumerate(coeffs):
            y += c * (x ** power)
            
        plt.plot(x, y, label=f"{labels[layer_idx]}", color=colors.get(layer_idx, 'black'), linewidth=2.5)

    plt.title("TaylorKAN Polynomial Evolution (Version 1)", fontsize=14, fontweight='bold')
    plt.xlabel("Input Value (Normalized)", fontsize=12)
    plt.ylabel("Output Response", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=300)
    print(f"\nGraph saved to: {output_path}")

# ---------------------------------------------------------
# 4. Execution
# ---------------------------------------------------------
def main():
    print(f"--- Starting Taylor Formula Extraction (A11 Version) ---")
    
    conf = load_config_from_file(CONFIG_PATH)
    
    # Override var_num from meta.json if possible
    detected_var_num = get_var_num_from_meta(META_JSON)
    if detected_var_num:
        print(f"Overriding var_num: {conf['var_num']} -> {detected_var_num}")
        conf['var_num'] = detected_var_num

    ModelClass = import_model_class()
    
    print("Building Model...")
    model = ModelClass(
        hist_len=conf["hist_len"], pred_len=conf["pred_len"], var_num=conf["var_num"],
        pv_index=conf.get("pv_index", 0), token_dim=conf.get("token_dim", 64),
        layer_num=conf.get("layer_num", 3), layer_hp=conf.get("layer_hp"),
        use_layernorm=conf.get("use_layernorm", True), dropout=conf.get("dropout", 0.1),
        fusion=conf.get("fusion", "mean"), baseline_mode=conf.get("baseline_mode", "last24_repeat"),
        use_pv_revin=conf.get("use_pv_revin", True), revin_eps=conf.get("revin_eps", 1e-5),
        enforce_nonneg=conf.get("enforce_nonneg", True),
    )

    print(f"Loading Checkpoint: {CKPT_PATH}")
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        state_dict = {k.replace("model.", ""): v for k, v in ckpt['state_dict'].items()}
        model.load_state_dict(state_dict, strict=False)
        model.to(DEVICE).eval()
    else:
        print(f"【エラー】Checkpointが見つかりません: {CKPT_PATH}")
        return

    results = {}
    
    # Extract from Layer 0 and Layer 2
    results[0] = analyze_taylor_layer_force(model, 0, expert_idx=2)
    results[2] = analyze_taylor_layer_force(model, 2, expert_idx=2)

    if any(v is not None for v in results.values()):
        plot_path = os.path.join(OUTPUT_DIR, "taylor_brake_evolution_A11.png")
        plot_polynomials(results, plot_path)
    else:
        print("No coefficients extracted.")

    print(f"--- Completed. Check folder: {OUTPUT_DIR} ---")

if __name__ == "__main__":
    main()