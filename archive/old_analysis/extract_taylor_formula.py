# -*- coding: utf-8 -*-
r"""
MMK TaylorKAN Formula Extractor V3 (Flexible Order Mode)
--------------------------------------------------------
Fix: Relaxed coefficient shape check to accept Order 3 (Size 4).
Target: Extract shape (64, 64, 4) -> Cubic Polynomial (x^0 to x^3).
"""

import os
import sys
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import torch

# ---------------------------------------------------------
# 1. User Environment Settings
# ---------------------------------------------------------
PROJECT_ROOT = os.getcwd()
CONFIG_PATH = os.path.join(PROJECT_ROOT, r"config\tsukuba_conf\MMK_FusionPV_Flatten_A9_96.py")
CKPT_PATH = os.path.join(PROJECT_ROOT, r"save\MMK_FusionPV_FeatureToken\version_0\checkpoints\last.ckpt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "research_formula_analysis")

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

def import_model_class():
    sys.path.append(PROJECT_ROOT)
    try:
        from easytsf.model.MMK_FusionPV_FeatureToken import MMK_FusionPV_FeatureToken
        return MMK_FusionPV_FeatureToken
    except ImportError:
        model_file = os.path.join(PROJECT_ROOT, "easytsf", "model", "MMK_FusionPV_FeatureToken.py")
        spec = importlib.util.spec_from_file_location("MMK_FusionPV_FeatureToken", model_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.MMK_FusionPV_FeatureToken

# ---------------------------------------------------------
# 3. Main Analysis Logic
# ---------------------------------------------------------
def analyze_taylor_layer_force(model, layer_idx, expert_idx=2):
    """Force extract coefficients from a specific expert index (Index 2 = TaylorKAN)."""
    print(f"\n--- Analyzing Layer {layer_idx}, Expert {expert_idx} (TaylorKAN) ---")
    
    if layer_idx >= len(model.blocks):
        print(f"Error: Layer {layer_idx} does not exist.")
        return None

    block = model.blocks[layer_idx]
    
    if not hasattr(block, 'mok'):
        print("Block does not have 'mok' attribute.")
        return None
    
    experts = block.mok.experts
    target_expert = experts[expert_idx]
    
    coeffs = None
    
    print("Searching for coefficients...")
    for name, param in target_expert.named_parameters():
        shape = tuple(param.shape)
        print(f"  - Found param: {name} | shape: {shape}")
        
        # 【修正】末尾の次元が 3〜10 なら「多項式の係数」とみなす
        # 今回は (64, 64, 4) なので param.shape[-1] == 4 でヒットする
        if len(shape) >= 2 and (3 <= shape[-1] <= 10):
            coeffs = param.detach().cpu().numpy()
            print(f"    -> HIT! Found polynomial coefficients (Size {shape[-1]}).")
            break
            
    if coeffs is None:
        print("Could not identify coefficient parameters automatically.")
        return None

    # 代表多項式（平均）の算出
    if coeffs.ndim == 3:
        # (Out, In, Order) -> (Order)
        mean_coeffs = np.mean(coeffs, axis=(0, 1))
    elif coeffs.ndim == 2:
        # (Dim, Order) -> (Order)
        mean_coeffs = np.mean(coeffs, axis=0)
    else:
        mean_coeffs = coeffs.flatten()

    print(f"Extracted Average Coefficients: {mean_coeffs}")
    
    # 数式の文字列化
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
    labels = {0: 'Layer 0 (Input)', 1: 'Layer 1', 2: 'Layer 2 (Deep Brake)'}

    for layer_idx, coeffs in layer_coeffs.items():
        if coeffs is None: continue
        
        y = np.zeros_like(x)
        for power, c in enumerate(coeffs):
            y += c * (x ** power)
            
        plt.plot(x, y, label=f"{labels[layer_idx]}", color=colors.get(layer_idx, 'black'), linewidth=2.5)

    plt.title("TaylorKAN Polynomial Evolution (Brake Verification)", fontsize=14, fontweight='bold')
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
    print(f"--- Starting Taylor Formula Extraction V3 ---")
    
    conf = load_config_from_file(CONFIG_PATH)
    ModelClass = import_model_class()
    
    # モデル構築
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
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state_dict = {k.replace("model.", ""): v for k, v in ckpt['state_dict'].items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE).eval()

    results = {}
    
    # Layer 0 と Layer 2 の Taylor (Index 2) を解析
    results[0] = analyze_taylor_layer_force(model, 0, expert_idx=2)
    results[2] = analyze_taylor_layer_force(model, 2, expert_idx=2)

    if any(v is not None for v in results.values()):
        plot_path = os.path.join(OUTPUT_DIR, "taylor_brake_evolution_v3.png")
        plot_polynomials(results, plot_path)
    else:
        print("No coefficients extracted.")

    print(f"--- Completed ---")

if __name__ == "__main__":
    main()