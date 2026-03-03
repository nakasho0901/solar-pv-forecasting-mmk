# -*- coding: utf-8 -*-
"""
make_error_plots_24_time.py
- make_error_plots_24.py を壊さずに、MMK_Mix_Time に対応した評価・可視化ツール（新規ファイル）
- pred.npy / true.npy / day_mask.npy を保存し、scatter / horizon別MAE/RMSE / サンプル時系列 を出力

使い方（例）
set PYTHONPATH=<YOUR_SOLAR_KAN_PATH> && python tools\\make_error_plots_24_time.py ^
  -c config\\tsukuba_conf\\MMK_Mix_PV_24_fromseries_norm_time.py ^
  --ckpt \"save\\...\\best-epoch....ckpt\" --split test ^
  --outdir \"save\\...\\error_plots\"
"""

import os
import sys
import argparse
import inspect

import numpy as np
import torch

import matplotlib.pyplot as plt

# ===== パス設定 =====
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../solar-kan
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from easytsf.runner.data_runner import NPYDataInterface


def get_model(model_name: str, exp_conf: dict):
    """
    configに書かれた引数だけをモデルに渡す（余計なキーで落ちないように）
    """
    if model_name == "MMK_Mix":
        from easytsf.model.MMK_Mix import MMK_Mix
        model_class = MMK_Mix
    elif model_name == "MMK_Mix_Time":
        from easytsf.model.MMK_Mix_Time import MMK_Mix_Time
        model_class = MMK_Mix_Time
    else:
        raise ValueError(f"[ERROR] Model {model_name} not supported in make_error_plots_24_time.py")

    sig = inspect.signature(model_class.__init__)
    valid_params = [p.name for p in sig.parameters.values() if p.name != "self"]
    filtered_conf = {k: v for k, v in exp_conf.items() if k in valid_params}
    return model_class(**filtered_conf)


def load_state_dict_to_model(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    # LightningModule の state_dict は "model.xxx" の形が多いので剥がす
    if any(k.startswith("model.") for k in sd.keys()):
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("model."):
                new_sd[k[len("model."):]] = v
        sd = new_sd

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[INFO] loaded ckpt: {ckpt_path}")
    if len(missing) > 0:
        print("[WARN] missing keys (top10):", missing[:10])
    if len(unexpected) > 0:
        print("[WARN] unexpected keys (top10):", unexpected[:10])


@torch.no_grad()
def run_inference(exp_conf: dict, ckpt_path: str, split: str, device: torch.device):
    """
    返り値:
      pred_np: (S, T, 1)
      true_np: (S, T, 1)
      day_np : (S, T, 1)  day=1 night=0
    """
    model = get_model(exp_conf["model_name"], exp_conf).to(device).eval()
    load_state_dict_to_model(model, ckpt_path, device)

    dm = NPYDataInterface(**exp_conf)
    # LightningDataModule 互換を想定
    try:
        dm.setup(stage="fit")
    except Exception:
        try:
            dm.setup()
        except Exception:
            pass

    if split == "train":
        loader = dm.train_dataloader()
    elif split == "val":
        loader = dm.val_dataloader()
    elif split == "test":
        loader = dm.test_dataloader()
    else:
        raise ValueError("--split must be train/val/test")

    pv_index = int(exp_conf.get("pv_index", 0))
    day_thr = float(exp_conf.get("day_mask_threshold", 0.5))

    preds = []
    trues = []
    days = []

    for batch in loader:
        # batch: var_x, marker_x, var_y, marker_y
        var_x, marker_x, var_y, marker_y = batch
        var_x = var_x.float().to(device)
        marker_x = marker_x.float().to(device)
        var_y = var_y.float().to(device)
        marker_y = marker_y.float().to(device)

        out = model(var_x, marker_x)
        pred = out[0] if isinstance(out, (tuple, list)) else out  # (B,T,C)

        # PVだけ取り出して (B,T,1)
        if pred.shape[-1] == 1:
            pred_pv = pred
            true_pv = var_y
        else:
            idx = pv_index if (0 <= pv_index < pred.shape[-1]) else 0
            pred_pv = pred[:, :, idx:idx + 1]
            true_pv = var_y[:, :, idx:idx + 1] if var_y.shape[-1] > 1 else var_y

        # day mask: marker_y は (B,T,1) を想定
        day = (marker_y > day_thr).float()
        if day.shape[-1] != 1:
            day = day[:, :, :1]

        preds.append(pred_pv.detach().cpu().numpy())
        trues.append(true_pv.detach().cpu().numpy())
        days.append(day.detach().cpu().numpy())

    pred_np = np.concatenate(preds, axis=0)
    true_np = np.concatenate(trues, axis=0)
    day_np = np.concatenate(days, axis=0)
    return pred_np, true_np, day_np


def mae(a, b):
    return np.mean(np.abs(a - b))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def safe_masked_mae(pred, true, mask):
    denom = np.maximum(mask.sum(), 1.0)
    return float(np.abs(pred - true).sum() / denom)


def safe_masked_rmse(pred, true, mask):
    denom = np.maximum(mask.sum(), 1.0)
    return float(np.sqrt(((pred - true) ** 2).sum() / denom))


def plot_scatter_day(true_np, pred_np, day_np, out_png):
    t = true_np.reshape(-1)
    p = pred_np.reshape(-1)
    d = day_np.reshape(-1) > 0.5
    t = t[d]
    p = p[d]

    plt.figure(figsize=(5, 5))
    plt.scatter(t, p, s=6, alpha=0.4)
    mn = float(min(t.min(), p.min())) if t.size else 0.0
    mx = float(max(t.max(), p.max())) if t.size else 1.0
    plt.plot([mn, mx], [mn, mx], linewidth=1)
    plt.xlabel("True (normalized)")
    plt.ylabel("Pred (normalized)")
    plt.title("Scatter (Day only)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_by_horizon(true_np, pred_np, day_np, out_png_mae_all, out_png_mae_day, out_png_rmse_all, out_png_rmse_day):
    S, T, _ = true_np.shape
    mae_all = []
    mae_day = []
    rmse_all = []
    rmse_day = []

    for h in range(T):
        t = true_np[:, h, 0]
        p = pred_np[:, h, 0]
        m = day_np[:, h, 0] > 0.5

        mae_all.append(float(np.mean(np.abs(p - t))))
        rmse_all.append(float(np.sqrt(np.mean((p - t) ** 2))))

        # day only
        if m.any():
            mae_day.append(float(np.mean(np.abs(p[m] - t[m]))))
            rmse_day.append(float(np.sqrt(np.mean((p[m] - t[m]) ** 2))))
        else:
            mae_day.append(np.nan)
            rmse_day.append(np.nan)

    x = np.arange(T)

    plt.figure(figsize=(7, 4))
    plt.plot(x, mae_all)
    plt.xlabel("Horizon (hour)")
    plt.ylabel("MAE (normalized)")
    plt.title("MAE by Horizon (All)")
    plt.tight_layout()
    plt.savefig(out_png_mae_all)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x, mae_day)
    plt.xlabel("Horizon (hour)")
    plt.ylabel("MAE (normalized)")
    plt.title("MAE by Horizon (Day)")
    plt.tight_layout()
    plt.savefig(out_png_mae_day)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x, rmse_all)
    plt.xlabel("Horizon (hour)")
    plt.ylabel("RMSE (normalized)")
    plt.title("RMSE by Horizon (All)")
    plt.tight_layout()
    plt.savefig(out_png_rmse_all)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(x, rmse_day)
    plt.xlabel("Horizon (hour)")
    plt.ylabel("RMSE (normalized)")
    plt.title("RMSE by Horizon (Day)")
    plt.tight_layout()
    plt.savefig(out_png_rmse_day)
    plt.close()


def plot_timeseries_samples(true_np, pred_np, day_np, outdir, n=6):
    S, T, _ = true_np.shape
    n = min(n, S)
    for i in range(n):
        t = true_np[i, :, 0]
        p = pred_np[i, :, 0]
        m = day_np[i, :, 0]

        plt.figure(figsize=(9, 3.2))
        plt.plot(t, label="true")
        plt.plot(p, label="pred")
        # day 영역の目印（薄い縦線）
        for h in range(T):
            if m[h] < 0.5:
                plt.axvline(h, alpha=0.06)

        plt.title(f"timeseries_sample{i:02d} (gray lines=night)")
        plt.xlabel("hour")
        plt.ylabel("normalized PV")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(outdir, f"timeseries_sample{i:02d}.png")
        plt.savefig(out)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # config load
    import importlib.util
    spec = importlib.util.spec_from_file_location("config_module", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    exp_conf = config_module.exp_conf

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    pred_np, true_np, day_np = run_inference(exp_conf, args.ckpt, args.split, device=device)

    # save arrays
    np.save(os.path.join(args.outdir, "pred.npy"), pred_np)
    np.save(os.path.join(args.outdir, "true.npy"), true_np)
    np.save(os.path.join(args.outdir, "day_mask.npy"), day_np)

    # metrics
    mae_all = mae(pred_np, true_np)
    rmse_all = rmse(pred_np, true_np)
    mse_all = float(np.mean((pred_np - true_np) ** 2))

    # day only
    d = day_np > 0.5
    mae_day = safe_masked_mae(pred_np, true_np, d)
    rmse_day = safe_masked_rmse(pred_np, true_np, d)
    mse_day = float(((pred_np - true_np) ** 2 * d).sum() / max(d.sum(), 1.0))

    print("\n=== Recomputed metrics (normalized) ===")
    print(f"MAE(all)  = {mae_all:.6f} [PV]")
    print(f"RMSE(all) = {rmse_all:.6f} [PV]")
    print(f"MSE(all)  = {mse_all:.6f} [PV^2]")
    print(f"MAE(day)  = {mae_day:.6f} [PV]")
    print(f"RMSE(day) = {rmse_day:.6f} [PV]")
    print(f"MSE(day)  = {mse_day:.6f} [PV^2]\n")

    # plots
    plot_scatter_day(true_np, pred_np, day_np, os.path.join(args.outdir, "scatter_day.png"))

    plot_by_horizon(
        true_np, pred_np, day_np,
        os.path.join(args.outdir, "mae_by_horizon_all.png"),
        os.path.join(args.outdir, "mae_by_horizon_day.png"),
        os.path.join(args.outdir, "rmse_by_horizon_all.png"),
        os.path.join(args.outdir, "rmse_by_horizon_day.png"),
    )

    plot_timeseries_samples(true_np, pred_np, day_np, args.outdir, n=6)

    print(f"[INFO] saved plots & arrays to: {args.outdir}")


if __name__ == "__main__":
    main()
