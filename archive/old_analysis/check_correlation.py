import numpy as np
import pandas as pd
import os

data_path = "dataset_PV/processed_hardzero/train_x.npy"

def check_correlation():
    if not os.path.exists(data_path):
        print("データが見つかりません。")
        return

    # データをロード
    data = np.load(data_path)
    
    # 全データを平坦化して、変数間の相関を計算
    flattened = data.reshape(-1, data.shape[2])
    df = pd.DataFrame(flattened)
    corr = df.corr()
    
    print("\n--- 変数 0 (発電量) との相関係数 ---")
    for i in range(len(corr)):
        print(f"変数 {i} との関係: {corr[0][i]:.4f}")
    
    print("\n※ 1.0 に近いほど、その変数が発電量と連動しています（日射量）。")
    print("※ 0.0 に近いほど、発電量とは無関係なデータです。")

if __name__ == "__main__":
    check_correlation()