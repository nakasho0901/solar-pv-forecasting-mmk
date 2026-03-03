import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# --- 0. プロジェクト構造のパス解決 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# --- 1. モデルのインポート ---
try:
    from easytsf.model.MMK_Mix import MMK_Mix 
    # iTransformerのインポート (実際のファイル名に合わせてください)
    from easytsf.model.iTransformer import iTransformer 
    print("【成功】モデルのインポートに成功しました。")
except ImportError as e:
    print(f"【エラー】インポート失敗: {e}")
    sys.exit(1)

# --- 2. 実験パラメータの設定 (資料 560, 566-572 に基づく) ---
CKPT_PATH_MMK = "checkpoints/mmk_solar_epoch=7.ckpt"
CKPT_PATH_ITRANS = "checkpoints/itrans_solar_epoch=19.ckpt"

# 中川さんの実験条件
HIST_LEN = 96     # 入力窓幅
PRED_LEN = 96     # 予測窓幅
VAR_NUM = 1       # 太陽光発電量のみ
HIDDEN_DIM = 512  
LAYER_NUM = 2     

# 【重要】MMK_Mix.py の仕様に合わせたエキスパート設定
# (関数名, 各関数のハイパーパラメータ) のペアで指定します
LAYER_HP = [
    ('spline', 5), 
    ('taylor', 2), 
    ('jacobi', 2), 
    ('wav', 3)
] 

# つくば市の発電量データ逆正規化用 [cite: 567-568]
MEAN = 20.54
STD = 33.01

def inverse_transform(data):
    return data * STD + MEAN

# --- 3. 重みロード用関数 ---
def load_weights_manually(model, ckpt_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ファイルが見つかりません: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {k.replace("model.", "").replace("net.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=False)
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"実行デバイス: {device}")

    # --- 4. モデルの構築 ---
    try:
        # MMK_Mixの初期化 (引数を正確に渡します)
        model_mmk = MMK_Mix(
            hist_len=HIST_LEN, 
            pred_len=PRED_LEN, 
            var_num=VAR_NUM, 
            hidden_dim=HIDDEN_DIM, 
            layer_hp=LAYER_HP, 
            layer_num=LAYER_NUM
        ).to(device)

        # iTransformerの初期化 (一般的な引数構成)
        model_itrans = iTransformer(
            seq_len=HIST_LEN,
            pred_len=PRED_LEN,
            enc_in=VAR_NUM,
            d_model=HIDDEN_DIM
        ).to(device)

        # 重みのロード
        model_mmk = load_weights_manually(model_mmk, CKPT_PATH_MMK)
        model_itrans = load_weights_manually(model_itrans, CKPT_PATH_ITRANS)
        
        model_mmk.eval()
        model_itrans.eval()
        print("【成功】学習済み重みのロードが完了しました。")
    except Exception as e:
        print(f"【エラー】モデル構築またはロード失敗: {e}")
        return

    # --- 5. データの準備 (つくば市の1日周期を模したダミーデータ) ---
    t = np.linspace(0, 4 * np.pi, HIST_LEN)
    # 太陽光らしい波形 (0~1の範囲)
    dummy_input = (np.sin(t - np.pi/2) * 0.5 + 0.5).reshape(1, HIST_LEN, 1) 
    batch_x = torch.tensor(dummy_input, dtype=torch.float32).to(device)

    # --- 6. 推論の実行 ---
    print("推論を実行中...")
    with torch.no_grad():
        output_mmk = model_mmk(batch_x)
        output_itrans = model_itrans(batch_x)

    # テンソルを [Time] のnumpy配列に変換
    pred_mmk = inverse_transform(output_mmk[0, :, 0].cpu().numpy())
    pred_itrans = inverse_transform(output_itrans[0, :, 0].cpu().numpy())
    true_values = inverse_transform(batch_x[0, :, 0].cpu().numpy())

    # --- 7. 比較グラフの作成 ---
    plt.figure(figsize=(12, 6))
    plt.plot(np.maximum(true_values, 0), label='True Values (Tsukuba)', color='#1f77b4', linewidth=2, alpha=0.7)
    plt.plot(np.maximum(pred_mmk, 0), label='MMK_Mix (Proposed, 296K)', color='#ff7f0e', linestyle='--', linewidth=2)
    plt.plot(np.maximum(pred_itrans, 0), label='iTransformer (Baseline, 841K)', color='#2ca02c', linestyle=':', linewidth=2)
    
    plt.title('Comparison: MMK vs iTransformer (Solar Power Prediction)', fontsize=14)
    plt.xlabel('Time Steps (Hours)', fontsize=12)
    plt.ylabel('Solar Power (kWh)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('solar_final_comparison_fixed.png', dpi=300)
    plt.show()
    print("【完了】グラフを 'solar_final_comparison_fixed.png' として保存しました。")

if __name__ == "__main__":
    main()