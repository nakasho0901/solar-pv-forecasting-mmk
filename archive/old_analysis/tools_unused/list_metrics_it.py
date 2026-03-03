# tools/list_metrics_it.py
# iTransformer の各実験フォルダから metrics.csv を読み取り、
# test/mae（あれば）や val_loss の最終値、最新ckptのパスを一覧表示します。

import os, csv, glob

BASE = r"save\iTransformer_Tsukuba"
EXPS = [d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d)) and d.lower() != "preds"]
EXPS.sort()

def last_row_dict(csv_path):
    if not os.path.exists(csv_path):
        return None
    last = None
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = row
    return last

def pick(d, keys):
    for k in keys:
        if d is not None and k in d and d[k] not in (None, ""):
            return d[k]
    return None

print("exp_hash | test_mae | val_loss | latest_ckpt")
print("-"*80)
for exp in EXPS:
    seed_dir = os.path.join(BASE, exp, "seed_1")
    mpath = os.path.join(seed_dir, "metrics.csv")
    row = last_row_dict(mpath)
    test_mae = pick(row, ["test/mae","test_mae","mae"])  # 環境で列名が異なる想定
    val_loss = pick(row, ["val/loss","val_loss","val/mae","val_mae"])

    ckpts = glob.glob(os.path.join(seed_dir, "checkpoints", "*.ckpt"))
    latest_ckpt = max(ckpts, key=os.path.getmtime) if ckpts else ""

    print(f"{exp} | {test_mae} | {val_loss} | {latest_ckpt}")
