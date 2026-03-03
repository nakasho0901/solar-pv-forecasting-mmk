import torch
import numpy as np

def analyze_taylor_real():
    path = "save/MMK_FusionPV_FeatureToken/version_12/checkpoints/last.ckpt"
    ckpt = torch.load(path, map_location='cpu')
    sd = ckpt['state_dict']

    for l in range(3):
        key = f"model.blocks.{l}.mok.experts.2.transform.coeffs"
        if key in sd:
            coeffs = sd[key] # Shape: (64, 64, 4)
            
            # 絶対値の平均（各項がどれくらい「力」を持っているか）
            abs_mean = coeffs.abs().mean(dim=(0, 1)).numpy()
            
            print(f"\n[Layer {l} Taylorの実力（平均パワー）]")
            print(f"  定数項の平均影響度: {abs_mean[0]:.4f}")
            print(f"  1次項(x)の平均影響度: {abs_mean[1]:.4f}")
            print(f"  2次項(x^2)の平均影響度: {abs_mean[2]:.4f}")
            print(f"  3次項(x^3)の平均影響度: {abs_mean[3]:.4f}")

            # 実際の多項式のサンプル（最初の1つ）
            c = coeffs[0, 0].numpy()
            print(f"  代表的な1つの式: f(x) = {c[0]:.4f} {'+' if c[1]>=0 else '-'} {abs(c[1]):.4f}x {'+' if c[2]>=0 else '-'} {abs(c[2]):.4f}x^2 {'+' if c[3]>=0 else '-'} {abs(c[3]):.4f}x^3")

if __name__ == "__main__":
    analyze_taylor_real()