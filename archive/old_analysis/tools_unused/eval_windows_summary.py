# tools/eval_windows_summary.py
# テスト集合の全ウィンドウで MMK と iTransformer（など2モデル）の MAE/MSE を集計し、
# CSVと各種グラフ（ヒスト、ECDF、散布図）を保存する。

import argparse, glob, os, sys, time, traceback
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from train import load_config  # noqa: E402
from easytsf.runner.data_runner import DataInterface  # noqa: E402
from easytsf.runner.exp_runner import LTSFRunner  # noqa: E402


def ensure_defaults(conf: dict) -> dict:
    d = dict(conf)
    d.setdefault("data_root", "dataset")
    d.setdefault("save_root", "save")
    d.setdefault("devices", "0,")
    d.setdefault("use_wandb", 0)
    d.setdefault("seed", 1)
    d.setdefault("num_workers", 0)
    return d


def find_latest_ckpt(save_root, model_name, dataset_name):
    tag = f"{model_name}_{dataset_name}"
    pattern = os.path.join(save_root, tag, "*", "seed_1", "checkpoints", "*.ckpt")
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"checkpoint not found: {pattern}")
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[0]


def load_runner(conf_path: str):
    conf_all = load_config(conf_path)
    tr_conf = conf_all[0] if isinstance(conf_all, tuple) else conf_all
    conf = ensure_defaults(tr_conf)
    ckpt = find_latest_ckpt(conf["save_root"], conf["model_name"], conf["dataset_name"])
    runner = LTSFRunner.load_from_checkpoint(ckpt, **conf)
    runner.eval(); runner.freeze()
    return conf, runner


def split_batch(batch):
    if isinstance(batch, dict):
        var_x = batch.get("var_x") or batch.get("x") or batch.get("inputs")
        marker_x = batch.get("marker_x") or batch.get("time_feat") or batch.get("time_feature")
        y_true = batch.get("y") or batch.get("label") or batch.get("target")
    else:
        var_x, marker_x, y_true = batch[0], batch[1], batch[2]
    return var_x, marker_x, y_true


def to_1d(a, var_idx: int):
    arr = a.detach().cpu().numpy()
    if arr.ndim == 3:   # [B, L, N]
        return arr[:, :, var_idx]
    elif arr.ndim == 2: # [B, L]
        return arr
    else:
        raise ValueError(f"unexpected shape {arr.shape}")


def ecdf(y):
    xs = np.sort(y)
    ys = np.arange(1, xs.size + 1) / xs.size
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c1", "--config1", required=True, help="モデル1（例: MMK）の設定.py")
    ap.add_argument("-c2", "--config2", required=True, help="モデル2（例: iTransformer）の設定.py")
    ap.add_argument("--var_idx", type=int, default=0, help="評価する変数インデックス（多変量時のみ）")
    ap.add_argument("--limit", type=int, default=None, help="先頭から何ウィンドウ評価するか（デバッグ用）")
    ap.add_argument("--out_dir", default="results", help="出力先ディレクトリ")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        conf1, runner1 = load_runner(args.config1)
        conf2, runner2 = load_runner(args.config2)
        dataset_name = conf1["dataset_name"]
        model1 = conf1["model_name"]
        model2 = conf2["model_name"]

        # データは conf1 に合わせる（同一データセット前提）
        dm = DataInterface(**conf1)
        if hasattr(dm, "setup"):
            try: dm.setup("test")
            except TypeError: dm.setup(stage=None)
        loader = dm.test_dataloader()

        maes1, mses1, maes2, mses2 = [], [], [], []
        gidx = 0

        with torch.no_grad():
            for batch in loader:
                var_x, marker_x, y_true = split_batch(batch)
                var_x = var_x.float(); marker_x = marker_x.float(); y_true = y_true.float()

                y_pred1 = runner1.model(var_x, marker_x) if hasattr(runner1, "model") else runner1(var_x, marker_x)
                y_pred2 = runner2.model(var_x, marker_x) if hasattr(runner2, "model") else runner2(var_x, marker_x)

                # 2Dに揃える：[B, pred_len]
                true_2d = to_1d(y_true, args.var_idx)
                pred1_2d = to_1d(y_pred1, args.var_idx)
                pred2_2d = to_1d(y_pred2, args.var_idx)

                # バッチ内の各サンプルでMAE/MSE
                for b in range(true_2d.shape[0]):
                    t = true_2d[b]
                    p1 = pred1_2d[b]
                    p2 = pred2_2d[b]
                    maes1.append(np.mean(np.abs(p1 - t)))
                    mses1.append(np.mean((p1 - t) ** 2))
                    maes2.append(np.mean(np.abs(p2 - t)))
                    mses2.append(np.mean((p2 - t) ** 2))
                    gidx += 1
                    if args.limit is not None and gidx >= args.limit:
                        break
                if args.limit is not None and gidx >= args.limit:
                    break

        maes1 = np.asarray(maes1); mses1 = np.asarray(mses1)
        maes2 = np.asarray(maes2); mses2 = np.asarray(mses2)
        n = len(maes1)

        # サマリ
        def summary(x):  # mean/median/90th/95th
            return dict(mean=float(np.mean(x)),
                        median=float(np.median(x)),
                        p90=float(np.percentile(x, 90)),
                        p95=float(np.percentile(x, 95)))
        sum_mae1, sum_mse1 = summary(maes1), summary(mses1)
        sum_mae2, sum_mse2 = summary(maes2), summary(mses2)
        win_mae = float(np.mean(maes1 < maes2))   # モデル1の勝率（MAE）
        win_mse = float(np.mean(mses1 < mses2))   # モデル1の勝率（MSE）

        print(f"[SUMMARY] windows={n}, dataset={dataset_name}")
        print(f"[SUMMARY] {model1}  MAE: {sum_mae1} | MSE: {sum_mse1}")
        print(f"[SUMMARY] {model2}  MAE: {sum_mae2} | MSE: {sum_mse2}")
        print(f"[SUMMARY] WinRate ({model1} better): MAE={win_mae:.3f}, MSE={win_mse:.3f}")

        # CSV出力
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"{dataset_name}_{model1}_vs_{model2}_{ts}"
        csv_path = os.path.join(args.out_dir, f"per_window_metrics_{base}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("global_idx,mae_"+model1+",mse_"+model1+",mae_"+model2+",mse_"+model2+"\n")
            for i in range(n):
                f.write(f"{i},{maes1[i]:.6f},{mses1[i]:.6f},{maes2[i]:.6f},{mses2[i]:.6f}\n")
        print(f"[SAVE] {csv_path}")

        # 図1: ヒストグラム（MAE/MSE）
        plt.figure(figsize=(10,4))
        bins = max(20, int(np.sqrt(n)))
        plt.hist(maes1, bins=bins, alpha=0.5, label=f"{model1} MAE")
        plt.hist(maes2, bins=bins, alpha=0.5, label=f"{model2} MAE")
        plt.title(f"Per-window MAE Histogram ({dataset_name})")
        plt.xlabel("MAE"); plt.ylabel("Count"); plt.legend(); plt.tight_layout()
        p_hist_mae = os.path.join(args.out_dir, f"hist_mae_{base}.png"); plt.savefig(p_hist_mae, dpi=150)

        plt.figure(figsize=(10,4))
        plt.hist(mses1, bins=bins, alpha=0.5, label=f"{model1} MSE")
        plt.hist(mses2, bins=bins, alpha=0.5, label=f"{model2} MSE")
        plt.title(f"Per-window MSE Histogram ({dataset_name})")
        plt.xlabel("MSE"); plt.ylabel("Count"); plt.legend(); plt.tight_layout()
        p_hist_mse = os.path.join(args.out_dir, f"hist_mse_{base}.png"); plt.savefig(p_hist_mse, dpi=150)

        # 図2: ECDF（累積分布）
        plt.figure(figsize=(10,4))
        x1,y1 = ecdf(maes1); x2,y2 = ecdf(maes2)
        plt.plot(x1,y1,label=f"{model1} MAE")
        plt.plot(x2,y2,label=f"{model2} MAE")
        plt.title(f"ECDF of Per-window MAE ({dataset_name})")
        plt.xlabel("MAE"); plt.ylabel("Cumulative prob."); plt.legend(); plt.tight_layout()
        p_ecdf_mae = os.path.join(args.out_dir, f"ecdf_mae_{base}.png"); plt.savefig(p_ecdf_mae, dpi=150)

        plt.figure(figsize=(10,4))
        x1,y1 = ecdf(mses1); x2,y2 = ecdf(mses2)
        plt.plot(x1,y1,label=f"{model1} MSE")
        plt.plot(x2,y2,label=f"{model2} MSE")
        plt.title(f"ECDF of Per-window MSE ({dataset_name})")
        plt.xlabel("MSE"); plt.ylabel("Cumulative prob."); plt.legend(); plt.tight_layout()
        p_ecdf_mse = os.path.join(args.out_dir, f"ecdf_mse_{base}.png"); plt.savefig(p_ecdf_mse, dpi=150)

        # 図3: 散布図（各窓の MAE: model1 vs model2）
        plt.figure(figsize=(5,5))
        plt.scatter(maes1, maes2, s=6, alpha=0.6)
        lim = [0, max(np.max(maes1), np.max(maes2))*1.05]
        plt.plot(lim, lim, linestyle="--")  # y=x
        plt.xlim(lim); plt.ylim(lim)
        plt.xlabel(f"{model1} MAE"); plt.ylabel(f"{model2} MAE")
        plt.title(f"Per-window MAE Scatter ({dataset_name})")
        plt.tight_layout()
        p_scatter_mae = os.path.join(args.out_dir, f"scatter_mae_{base}.png"); plt.savefig(p_scatter_mae, dpi=150)

        print(f"[SAVE] {p_hist_mae}\n[SAVE] {p_hist_mse}\n[SAVE] {p_ecdf_mae}\n[SAVE] {p_ecdf_mse}\n[SAVE] {p_scatter_mae}")

    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    main()
