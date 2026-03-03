# tools/plot_pred_vs_actual.py
# 直近チェックポイントから復元し、「予測 vs 実測」を保存。
# 実測=青、予測=オレンジ点線。--hours で表示長を指定。

import argparse
import glob
import os
import sys
import time
import traceback

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --- プロジェクトルートを import パスに追加 ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from train import load_config  # noqa: E402
from easytsf.runner.data_runner import DataInterface  # noqa: E402
from easytsf.runner.exp_runner import LTSFRunner  # noqa: E402


def log(msg: str):
    print(f"[PLOT] {msg}", flush=True)


def find_latest_ckpt(save_dir: str, model_name: str, dataset_name: str = "Tsukuba") -> str:
    tag = f"{model_name}_{dataset_name}"
    pattern = os.path.join(save_dir, tag, "*", "seed_1", "checkpoints", "*.ckpt")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"チェックポイントが見つかりません: {pattern}")
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="実験設定ファイル（.py）")
    ap.add_argument("--save_dir", default="save", help="チェックポイント親ディレクトリ")
    ap.add_argument("--out_dir", default="results", help="図の保存先")
    ap.add_argument("--sample_idx", type=int, default=0, help="バッチ内サンプル番号（0始まり）")
    ap.add_argument("--var_idx", type=int, default=0, help="変数インデックス（0始まり）")
    ap.add_argument("--hours", type=int, default=24, help="描画する先の時間数（例: 24）")
    args = ap.parse_args()

    try:
        os.makedirs(args.out_dir, exist_ok=True)
        log(f"ROOT={ROOT}")
        log(f"config={args.config}")

        # ---- 設定の読み込み ----
        conf_all = load_config(args.config)
        training_conf = conf_all[0] if isinstance(conf_all, tuple) else conf_all
        conf = dict(training_conf)

        # ---- train.py 側デフォルトを補完 ----
        defaults = dict(
            data_root="dataset",
            save_root="save",
            devices="0,",
            use_wandb=0,
            seed=1,
            num_workers=0,
        )
        for k, v in defaults.items():
            conf.setdefault(k, v)

        log(f"model={conf.get('model_name')} dataset={conf.get('dataset_name')} "
            f"hist_len={conf.get('hist_len')} pred_len={conf.get('pred_len')}")
        log(f"data_root={conf.get('data_root')} save_root={conf.get('save_root')}")

        # ---- DataModule 準備（テスト）----
        data_module = DataInterface(**conf)
        if hasattr(data_module, "setup"):
            try:
                data_module.setup("test")
            except TypeError:
                data_module.setup(stage=None)
        test_loader = data_module.test_dataloader()
        log("test_loader 準備完了")

        # ---- 最新 ckpt 探索 ----
        ckpt_path = find_latest_ckpt(conf["save_root"], conf["model_name"], conf["dataset_name"])
        log(f"ckpt={ckpt_path}")

        # ---- モデル復元 ----
        runner = LTSFRunner.load_from_checkpoint(ckpt_path, **conf)
        runner.eval()
        runner.freeze()
        log("model 復元完了")

        # ---- 推論 ----
        batch = next(iter(test_loader))
        if isinstance(batch, dict):
            var_x = batch.get("var_x") or batch.get("x") or batch.get("inputs")
            marker_x = batch.get("marker_x") or batch.get("time_feat") or batch.get("time_feature")
            y_true = batch.get("y") or batch.get("label") or batch.get("target")
        else:
            if len(batch) < 3:
                raise RuntimeError("バッチの形が想定外（少なくとも var_x, marker_x, y が必要）")
            var_x, marker_x, y_true = batch[0], batch[1], batch[2]

        # ★ 型を float32 に統一（iTransformer などで必須）
        if isinstance(var_x, torch.Tensor):   var_x = var_x.float()
        if isinstance(marker_x, torch.Tensor): marker_x = marker_x.float()
        if isinstance(y_true, torch.Tensor):  y_true = y_true.float()

        log(f"batch shapes: var_x={tuple(var_x.shape)}, marker_x={tuple(marker_x.shape)}, y={tuple(y_true.shape)}")

        with torch.no_grad():
            if hasattr(runner, "model"):
                y_pred = runner.model(var_x, marker_x)
            else:
                y_pred = runner(var_x, marker_x)
        log(f"pred shape: {tuple(y_pred.shape)}")

        # ---- 配列整形 ----
        y_pred_np = to_np(y_pred)
        y_true_np = to_np(y_true)
        b = int(args.sample_idx)
        v = int(args.var_idx)

        # 単変量/多変量に対応
        if y_pred_np.ndim == 2:  # [B, pred_len]
            pred_1d = y_pred_np[b]
            true_1d = y_true_np[b]
        else:  # [B, pred_len, N]
            pred_1d = y_pred_np[b, :, v]
            true_1d = y_true_np[b, :, v]

        # 先の hours のみを描画
        clip = int(args.hours)
        clip = max(1, min(clip, pred_1d.shape[0]))
        pred_1d = pred_1d[:clip]
        true_1d = true_1d[:clip]
        x = np.arange(clip)

        # ---- プロット ----
        plt.figure(figsize=(10, 4))
        plt.plot(x, true_1d, label="True Values")                   # 青（デフォ）
        plt.plot(x, pred_1d, label="Predicted Values", linestyle="--")  # オレンジ点線（デフォ）
        plt.title(f"{conf['model_name']}: Last {clip} Hours Forecast")
        plt.xlabel("Time Steps (Hours)")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(
            args.out_dir,
            f"pred_last{clip}h_{conf['model_name']}_{conf['dataset_name']}_b{b}_v{v}_{ts}.png",
        )
        plt.savefig(out_path, dpi=150)
        log(f"SAVED: {out_path}")

    except Exception:
        log("エラー発生。詳細スタックトレース：")
        traceback.print_exc()


if __name__ == "__main__":
    main()
