# -*- coding: utf-8 -*-
import torch
import pandas as pd
import os

# 解析するチェックポイントの絶対パスを指定（確実にするため）
CKPT_PATH = r"C:\Users\nakas\my_project\solar-kan\save\MMK_Mix_PV_96\MMK_Mix_PV_96\seed_2025\version_1\checkpoints\epoch=epoch=1-step=step=722.ckpt"
SAVE_NAME = "kan_symbolic_formulas.csv"

def extract_formulas():
    if not os.path.exists(CKPT_PATH):
        print(f"【エラー】チェックポイントが見つかりません: {CKPT_PATH}")
        return

    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    state_dict = ckpt['state_dict']
    results = []

    print("【解析中】モデルの重みをスキャンしています...")

    for layer in range(3):
        # Taylor項の抽出
        key_taylor = f'model.layers.{layer}.mok.experts.2.transform.coeffs'
        if key_taylor in state_dict:
            c = state_dict[key_taylor].mean(dim=(0, 1)).numpy()
            results.append({
                "Layer": layer, "Type": "Taylor",
                "Formula": f"y = {c[0]:.4f} + {c[1]:.4f}x + {c[2]:.4f}x^2 + {c[3]:.4f}x^3"
            })

        # Jacobi項の抽出
        key_jacobi = f'model.layers.{layer}.mok.experts.3.transform.jacobi_coeffs'
        if key_jacobi in state_dict:
            c = state_dict[key_jacobi].mean(dim=(0, 1)).numpy()
            formula = " + ".join([f"{val:.4f}P_{i}(x)" for i, val in enumerate(c)])
            results.append({"Layer": layer, "Type": "Jacobi", "Formula": f"y = {formula}"})

    if not results:
        print("【警告】数式データが1件も抽出されませんでした。モデルのキー名を確認してください。")
        return

    df = pd.DataFrame(results)
    # 絶対パスで保存場所を指定
    save_path = os.path.join(os.getcwd(), SAVE_NAME)
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    
    print("-" * 30)
    print(f"【成功】数式データを保存しました！")
    print(f"保存先: {save_path}")
    print("-" * 30)
    print(df[["Layer", "Type", "Formula"]]) # 画面にも内容を表示

if __name__ == "__main__":
    extract_formulas()