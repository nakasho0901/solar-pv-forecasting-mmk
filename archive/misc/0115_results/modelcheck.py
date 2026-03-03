import torch
from traingraph import PeakRunner, get_model

# モデルをロード
ckpt_path = "save_mix/MMK_Mix_96_Final/MMK_Mix/version_0/checkpoints/best-epochepoch=25-val_lossval/loss=0.4605.ckpt"
# ※パスは実際の環境に合わせてください

checkpoint = torch.load(ckpt_path, map_location="cpu")
state_dict = checkpoint["state_dict"]

# 専門家（Expert）の名前をすべて書き出す
experts_found = [key for key in state_dict.keys() if "mok.experts" in key]

print(f"【確認結果】")
print(f"見つかった専門家の数: {len(set([k.split('experts.')[1].split('.')[0] for k in experts_found]))}種類")
for k in sorted(list(set([k.split('experts.')[1].split('.')[0] for k in experts_found]))):
    print(f" - Expert {k} が存在します")