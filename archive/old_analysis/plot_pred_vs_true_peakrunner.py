# -*- coding: utf-8 -*-
r"""
予測 vs 実測の可視化（1窓ぶん）

- traingraph.py と同じ PeakRunner / NPYDataInterface を使って推論するので、
  学習時と同じ前処理（ghi_boost 等）・同じ marker_x を使える。
- split（train/val/test）と window_index を指定して、1サンプルの予測を描画する。
- window_index は負数対応: -1=最後, -2=最後から2番目 ...

使い方例:
python tools\\plot_pred_vs_true_peakrunner.py -c config\\tsukuba_conf\\MMK_FusionPV_RevIN_96_8.py --ckpt "save\\MMK_Mix\\version_0\\checkpoints\\last.ckpt" --split test --window_index -1 --outdir results_predplots --device cpu
"""

import os
import sys
import argparse
import importlib.util
import numpy as np
import torch
import matplotlib.pyplot as plt

# ===== パス設定（念のため tools の1個上＝プロジェクトルートを sys.path に追加）=====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# traingraph.py の get_model / PeakRunner を再利用（学習と同条件にするため）
from traingraph import get_model, PeakRunner
from easytsf.runner.data_runner import NPYDataInterface


def load_config(config_path: str) -> dict:
    """configファイル（*.py）から exp_conf を読み込む"""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"[ERROR] config を読み込めません: {config_path}")
    spec.loader.exec_module(module)
    if not hasattr(module, "exp_conf"):
        raise ValueError(f"[ERROR] config に exp_conf がありません: {config_path}")
    return module.exp_conf


def get_dataloader(datamodule: NPYDataInterface, split: str):
    """datamodule から split に対応する dataloader を取る"""
    s = split.lower()
    if s == "train":
        return datamodule.train_dataloader()
    if s == "val":
        return datamodule.val_dataloader()
    if s == "test":
        return datamodule.test_dataloader()
    raise ValueError(f"[ERROR] split は train/val/test のどれかです: {split}")


def load_t0(data_dir: str, split: str):
    """split_t0.npy を読み込む（x/yと同じ順序のはず）"""
    p = os.path.join(data_dir, f"{split}_t0.npy")
    if not os.path.exists(p):
        return None
    return np.load(p)


def resolve_window_index_from_npy(data_dir: str, split: str, window_index: int) -> int:
    """
    window_index を解決する（datamodule.dataset を使わず npy から総数を取る）
    -1 は最後、-2 は最後から2番目...
    """
    x_path = os.path.join(data_dir, f"{split}_x.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"[ERROR] 見つかりません: {x_path}")

    # mmapでヘッダだけ読み、shape[0]（窓数）を取る
    x = np.load(x_path, mmap_mode="r")
    total = int(x.shape[0])

    idx = int(window_index)
    if idx < 0:
        idx = total + idx  # -1 -> total-1

    if idx < 0 or idx >= total:
        raise IndexError(f"[ERROR] window_index が範囲外です: {window_index} (resolved={idx}, total={total})")

    return idx


def fetch_one_sample_from_loader(loader, idx: int, batch_size: int):
    """
    DataLoader から idx 番目のサンプルを取り出す。
    loader は (var_x, marker_x, var_y, marker_y) を返す想定。
    """
    batch_idx = idx // batch_size
    inner_idx = idx % batch_size

    target_batch = None
    for b_i, batch in enumerate(loader):
        if b_i == batch_idx:
            target_batch = batch
            break

    if target_batch is None:
        raise IndexError(f"[ERROR] idx={idx} が DataLoader 範囲外です（batch_idx={batch_idx}）")

    var_x, marker_x, var_y, marker_y = target_batch
    return var_x[inner_idx:inner_idx + 1], marker_x[inner_idx:inner_idx + 1], var_y[inner_idx:inner_idx + 1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="config/tsukuba_conf/xxx.py")
    parser.add_argument("--ckpt", type=str, required=True, help="Lightning checkpoint path (*.ckpt)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--window_index", type=int, default=0, help="split内の窓index（負数OK: -1=最後）")
    parser.add_argument("--outdir", type=str, default="results_predplots", help="出力フォルダ")
    parser.add_argument("--device", type=str, default="cpu", help="cpu / cuda")
    args = parser.parse_args()

    exp_conf = load_config(args.config)

    # ===== datamodule 準備（学習時と同じ） =====
    datamodule = NPYDataInterface(**exp_conf)
    if hasattr(datamodule, "setup"):
        try:
            datamodule.setup(stage="fit")
        except TypeError:
            datamodule.setup()

    # ===== window_index 解決（-1対応、npyから総数を取る）=====
    data_dir = exp_conf["data_dir"]
    idx = resolve_window_index_from_npy(data_dir, args.split, args.window_index)

    # ===== dataloader =====
    loader = get_dataloader(datamodule, args.split)

    # ===== モデルを checkpoint から復元（学習時と同じ PeakRunner） =====
    base_model = get_model(exp_conf["model_name"], exp_conf)
    model_runner = PeakRunner.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        model=base_model,
        config=exp_conf,
        map_location=args.device
    )
    model_runner.eval()
    model_runner.to(args.device)

    # ===== 1サンプル取り出し =====
    batch_size = int(exp_conf.get("batch_size", 64))
    var_x1, marker_x1, var_y1 = fetch_one_sample_from_loader(loader, idx, batch_size)

    var_x1 = var_x1.float().to(args.device)
    marker_x1 = marker_x1.float().to(args.device)
    var_y1 = var_y1.float().to(args.device)

    # ===== 推論 =====
    with torch.no_grad():
        pred_y1 = model_runner._predict(var_x1, marker_x1)  # (1, pred_len, 1)

    pred = pred_y1.squeeze(0).squeeze(-1).detach().cpu().numpy()
    true = var_y1.squeeze(0).squeeze(-1).detach().cpu().numpy()

    # ===== 時間軸（t0があれば使う） =====
    t0 = load_t0(data_dir, args.split)
    if t0 is not None and len(t0) > idx:
        hist_len = int(exp_conf["hist_len"])
        pred_len = int(exp_conf["pred_len"])
        t0_i = np.array(t0[idx], dtype="datetime64[ns]")
        y_time = t0_i + (hist_len + np.arange(pred_len)) * np.timedelta64(1, "h")
        x_label = "time"
    else:
        y_time = np.arange(len(true))
        x_label = "step (h)"

    # ===== 保存先 =====
    os.makedirs(args.outdir, exist_ok=True)

    # ===== プロット（予測区間のみ） =====
    plt.figure()
    plt.plot(y_time, true, label="true")
    plt.plot(y_time, pred, label="pred")
    plt.xlabel(x_label)
    plt.ylabel("pv_kwh")
    plt.title(f"{exp_conf['model_name']} | {args.split} | window_index={args.window_index} (resolved={idx})")
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(args.outdir, f"pred_vs_true_{exp_conf['model_name']}_{args.split}_idx{idx}.png")
    plt.savefig(out_png, dpi=200)
    print(f"[OK] saved: {out_png}")


if __name__ == "__main__":
    main()
