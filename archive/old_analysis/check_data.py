import numpy as np
import os

# データパス（設定ファイルに合わせる）
data_path = "dataset_PV/processed_hardzero/train_x.npy"

if os.path.exists(data_path):
    data = np.load(data_path)
    print(f"データ形状: {data.shape}") # [サンプル数, 96, 8]
    
    # 最初のサンプルの各変数の平均値などを表示
    for i in range(data.shape[2]):
        col_data = data[:, :, i]
        print(f"変数 {i}: 平均={col_data.mean():.4f}, 最大={col_data.max():.4f}, 最小={col_data.min():.4f}")
    
    print("\n※通常、変数0が発電量、変数1付近が日射量(GHI)であることが多いです。")
else:
    print("データが見つかりません。パスを確認してください。")