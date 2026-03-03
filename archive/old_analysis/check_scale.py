# tools/check_scale.py
import numpy as np
import os

# 設定
DATA_PATH = "dataset_PV/processed_hardzero/train_y.npy"
RAW_PEAK_KW = 136.0  # あなたが知っている最大ピーク値

def check_scale():
    if not os.path.exists(DATA_PATH):
        print(f"ファイルが見つかりません: {DATA_PATH}")
        return

    # 正規化済みのデータを読み込む
    data = np.load(DATA_PATH)
    
    # データの統計量を計算
    val_min = np.min(data)
    val_max = np.max(data)
    
    print("--- 正規化済みデータの統計 ---")
    print(f"最小値 (夜間など): {val_min:.4f}")
    print(f"最大値 (ピーク):   {val_max:.4f}")
    
    # 夜間（生データ 0kW）が正規化後に val_min になっていると仮定
    # ピーク（生データ 136kW）が正規化後に val_max になっていると仮定
    # 公式: 生データ = (正規化値 * KW_FACTOR) + OFFSET
    
    # 2元連立方程式を解く:
    # 0   = (val_min * KW_FACTOR) + OFFSET
    # 136 = (val_max * KW_FACTOR) + OFFSET
    
    kw_factor = RAW_PEAK_KW / (val_max - val_min)
    offset = - (val_min * kw_factor)
    
    print("\n--- 算出された変換係数 ---")
    print(f"KW_FACTOR = {kw_factor:.2f}")
    print(f"OFFSET    = {offset:.2f}")
    print("\nこの数値を plot_prediction_kw.py の設定欄にコピーしてください。")

if __name__ == "__main__":
    check_scale()