import argparse
import os
import sys
import importlib.util

import numpy as np
import torch
import matplotlib.pyplot as plt

# easytsf をインポートできるようにパスを追加
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from easytsf.runner.data_runner import NPYDataInterface
from easytsf.runner.exp_runner import LTSFRunner


def load_exp_conf(config_path: str):
    """
    config/tsukuba_conf/xxx.py から exp_conf を読み込むヘルパー関数
    """
    spec = importlib.util.spec_from_file_location("cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    if not hasattr(cfg, "exp_conf"):
        raise ValueError(f"{config_path} に exp_conf が定義されていません。")
    return cfg.exp_conf


def main():
    parser = argparse.ArgumentParser(
        description="学習済み LTSFRunner (iTransformer / MMK など) のチェックポイントから "
                    "テストデータに対する予測を行い、MAE/MSE を計算して保存＆プロットするスクリプト"
    )
    parser.add_argument(
        "-c", "--config",
        required=True,
        help="実験設定ファイルへのパス（例: config/tsukuba_conf/iTransformer_PVProcessed_24for24.py）"
    )
    parser.add_argument(
        "-ckpt", "--checkpoint",
        required=True,
        help="学習済みチェックポイント (.ckpt) へのパス"
    )
    parser.add_argument(
        "-o", "--outdir",
        required=True,
        help="結果 (true_test.npy, pred_test.npy, PNG) を保存するディレクトリ"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="プロットに使うサンプル index（テストセット内の何番目か）"
    )

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. config から exp_conf をロード
    # ------------------------------------------------------------------
    exp_conf = load_exp_conf(args.config)

    # データローダ用の設定だけ取り出す（data_root, dataset_name, batch_size, num_workers があれば使う）
    data_keys = ["data_root", "dataset_name", "batch_size", "num_workers"]
    data_kwargs = {k: exp_conf[k] for k in data_keys if k in exp_conf}

    # ------------------------------------------------------------------
    # 2. DataModule (NPYDataInterface) 準備
    # ------------------------------------------------------------------
    dm = NPYDataInterface(**data_kwargs)
    dm.setup()
    test_loader = dm.test_dataloader()

    # ------------------------------------------------------------------
    # 3. モデルをチェックポイントからロード
    #    -> train.py と同じ LTSFRunner を使う
    # ------------------------------------------------------------------
    # ckpt 内に保存されている hparams を使いつつ、config 側の exp_conf も渡す
    # （exp_conf に含まれる data_root, dataset_name などもここで渡される）
    model = LTSFRunner.load_from_checkpoint(
        args.checkpoint,
        **exp_conf
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_true = []
    all_pred = []

    # ------------------------------------------------------------------
    # 4. テストデータに対して予測
    #    ※ train と同じように model(batch) で呼ぶのがポイント
    # ------------------------------------------------------------------
    with torch.no_grad():
        for batch in test_loader:
            # batch は [var_x, marker_x, var_y, marker_y] の 4 つ
            batch = [t.to(device).float() for t in batch]

            # LTSFRunner.forward(self, batch) の想定に合わせて「バッチ 1 個」を渡す
            pred = model(batch)

            # true（教師データ）は batch[2] = var_y
            var_y = batch[2]

            # 形状をそろえる: (B, T, 1, 1) -> (B, T, 1) に squeeze
            if pred.dim() == 4 and pred.size(-1) == 1:
                pred = pred.squeeze(-1)
            if var_y.dim() == 4 and var_y.size(-1) == 1:
                var_y = var_y.squeeze(-1)

            all_pred.append(pred.cpu().numpy())
            all_true.append(var_y.cpu().numpy())

    # ------------------------------------------------------------------
    # 5. Numpy にまとめて MAE/MSE を計算
    # ------------------------------------------------------------------
    y = np.concatenate(all_true, axis=0)   # shape: (N, out_len, 1) を想定
    pred = np.concatenate(all_pred, axis=0)

    # 念のため最後の次元 size=1 なら squeeze して (N, out_len, 1) に統一
    if y.ndim == 4 and y.shape[-1] == 1:
        y = y.squeeze(-1)
    if pred.ndim == 4 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)

    # ここまでで y, pred とも (N, out_len, 1) になっているはず
    # （もし (N, out_len) になっていてもブロードキャストで計算可能）

    err = pred - y
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))

    print("=== EVAL DONE ===")
    print("{")
    print(f'  "mae": {mae},')
    print(f'  "mse": {mse},')
    print(f'  "rmse": {rmse}')
    print("}")

    # .npy と JSON ライクなテキストも保存しておく
    true_path = os.path.join(args.outdir, "true_test.npy")
    pred_path = os.path.join(args.outdir, "pred_test.npy")
    np.save(true_path, y)
    np.save(pred_path, pred)

    summary_path = os.path.join(args.outdir, "metrics.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"mae={mae}\n")
        f.write(f"mse={mse}\n")
        f.write(f"rmse={rmse}\n")

    # ------------------------------------------------------------------
    # 6. 1サンプルだけ簡易プロット（オプション）
    # ------------------------------------------------------------------
    idx = max(0, min(args.index, y.shape[0] - 1))
    t = np.arange(y.shape[1])

    plt.figure(figsize=(10, 4))
    plt.plot(t, y[idx, :, 0], label="True")
    plt.plot(t, pred[idx, :, 0], label="Pred")
    plt.xlabel("Time step")
    plt.ylabel("PV output (normalized)")  # 正規化後の値（逆正規化していない場合）
    plt.title(f"Sample {idx} - MAE={mae:.3f}, MSE={mse:.3f}")
    plt.legend()
    plt.tight_layout()

    png_path = os.path.join(args.outdir, f"sample_{idx}.png")
    plt.savefig(png_path)
    plt.close()

    print(f"Saved:")
    print(f"  true_test.npy -> {true_path}")
    print(f"  pred_test.npy -> {pred_path}")
    print(f"  metrics.txt   -> {summary_path}")
    print(f"  sample plot   -> {png_path}")


if __name__ == "__main__":
    main()
