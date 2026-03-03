# -*- coding: utf-8 -*-
"""
tools/make_error_plots_24.py

目的:
- traingraph_24.py で学習した checkpoint を読み込み
- split(test/val/train) の予測を作成
- 誤差(MAE/MSE/RMSE)を再計算し、グラフ作成して保存する

出力(例):
- outdir/
  - pred.npy, true.npy, day_mask.npy
  - mae_by_horizon_all.png
  - mae_by_horizon_day.png
  - rmse_by_horizon_all.png
  - rmse_by_horizon_day.png
  - timeseries_sample00.png ... (n_plot枚)
  - scatter_day.png

誤差の単位:
- MAE: [PVの単位]（例: kW なら kW, Wh なら Wh）
- MSE: [PVの単位^2]
- RMSE: [PVの単位]
"""

import os
import argparse
import importlib.util
import inspect

import numpy as np
import torch
import matplotlib.pyplot as plt

from easytsf.runner.data_runner import NPYDataInterface


# ===== traingraph_24.py と同等の補助関数 =====
def _extract_pv_pred(pred: torch.Tensor, pv_index_if_multi: int = 0) -> torch.Tensor:
    """
    pred から PV 1本 (B,T,1) を取り出す。
    - pred が (B,T,1): そのまま
    - pred が (B,T,C>=2): pv_index_if_multi で選ぶ
    """
    if pred.dim() != 3:
        raise ValueError(f"[ERROR] pred shape unexpected: {tuple(pred.shape)} (expected 3D)")
    C = pred.shape[-1]
    if C == 1:
        return pred
    idx = int(pv_index_if_multi)
    if idx < 0 or idx >= C:
        idx = 0
    return pred[:, :, idx:idx + 1]


def _apply_ghi_boost(var_x: torch.Tensor, ghi_index: int = 2, factor: float = 1.0) -> torch.Tensor:
    """
    GHI を factor 倍する（必要な場合だけ）
    """
    if float(factor) == 1.0:
        return var_x
    x = var_x.clone()
    if x.dim() == 3 and x.shape[-1] > int(ghi_index):
        x[:, :, int(ghi_index)] = x[:, :, int(ghi_index)] * float(factor)
    return x


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    mask==1 の場所だけ平均（0除算回避つき）
    x, mask: (B,T,1)
    """
    denom = mask.sum().clamp_min(1.0)
    return (x * mask).sum() / denom


def get_model(model_name: str, exp_conf: dict):
    """
    config に書かれた引数だけをモデルに渡す（余計なキーで落ちないように）
    """
    if model_name in ["iTransformerPeak", "iTransformer"]:
        from easytsf.model.iTransformer_peak import iTransformerPeak
        model_class = iTransformerPeak
    elif model_name == "MMK_Mix":
        from easytsf.model.MMK_Mix import MMK_Mix
        model_class = MMK_Mix
    else:
        raise ValueError(f"[ERROR] Model {model_name} not supported.")

    sig = inspect.signature(model_class.__init__)
    valid_params = [p.name for p in sig.parameters.values() if p.name != "self"]
    filtered_conf = {k: v for k, v in exp_conf.items() if k in valid_params}
    return model_class(**filtered_conf)


@torch.no_grad()
def run_inference(exp_conf: dict, ckpt_path: str, split: str = "test", device: str = "cpu"):
    """
    checkpoint をロードして予測を作る。
    戻り値:
      pred_np: (N, T, 1)
      true_np: (N, T, 1)
      day_np : (N, T, 1)  0/1 の昼マスク
    """
    # config 由来の設定
    pv_index = int(exp_conf.get("pv_index", 0))
    ghi_index = int(exp_conf.get("ghi_index", 2))
    ghi_boost = float(exp_conf.get("ghi_boost", 1.0))
    day_thr = float(exp_conf.get("day_mask_threshold", 0.5))

    # モデル構築
    model = get_model(exp_conf["model_name"], exp_conf).to(device).eval()

    # Lightning の ckpt は "state_dict" のキーで保存されていることが多い
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    # traingraph_24.py は Runner24(LightningModule) を保存しているので
    # state_dict のキーに "model." が付いている想定で吸収する
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("model."):
            new_sd[k[len("model."):]] = v
        else:
            # もし runner ではなく model 直保存ならそのまま入る
            new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if len(unexpected) > 0:
        print("[WARN] unexpected keys:", unexpected[:10])
    if len(missing) > 0:
        print("[WARN] missing keys:", missing[:10])

    # データ
    dm = NPYDataInterface(**exp_conf)
    dm.setup(stage=None)

    if split == "test":
        loader = dm.test_dataloader()
    elif split == "val":
        loader = dm.val_dataloader()
    elif split == "train":
        loader = dm.train_dataloader()
    else:
        raise ValueError("split must be test/val/train")

    preds, trues, days = [], [], []

    for batch in loader:
        var_x, marker_x, var_y, marker_y = [_.float().to(device) for _ in batch]

        # 予測（traingraph_24.py と同じ流れ）
        x_in = _apply_ghi_boost(var_x, ghi_index=ghi_index, factor=ghi_boost)
        out = model(x_in, marker_x)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        pred_pv = _extract_pv_pred(pred, pv_index_if_multi=pv_index)

        day_mask = (marker_y > day_thr).float()

        preds.append(pred_pv.detach().cpu())
        trues.append(var_y.detach().cpu())
        days.append(day_mask.detach().cpu())

    pred_t = torch.cat(preds, dim=0)  # (N,T,1)
    true_t = torch.cat(trues, dim=0)  # (N,T,1)
    day_t  = torch.cat(days, dim=0)   # (N,T,1)

    return pred_t.numpy(), true_t.numpy(), day_t.numpy()


def compute_metrics(pred_np: np.ndarray, true_np: np.ndarray, day_np: np.ndarray):
    """
    全点(all) と 昼だけ(day) の
    - MAE [PV単位]
    - MSE [PV単位^2]
    - RMSE [PV単位]
    を計算する。
    """
    pred = torch.from_numpy(pred_np).float()
    true = torch.from_numpy(true_np).float()
    day  = torch.from_numpy(day_np).float()

    err = pred - true
    abs_err = err.abs()
    sq_err = err ** 2

    # 全点
    mae_all = abs_err.mean().item()
    mse_all = sq_err.mean().item()
    rmse_all = float(np.sqrt(mse_all))

    # 昼だけ（mask平均）
    mae_day = masked_mean(abs_err, day).item()
    mse_day = masked_mean(sq_err, day).item()
    rmse_day = float(np.sqrt(mse_day))

    # horizon別（Tごとの平均）
    # pred/true: (N,T,1) -> (T,)
    mae_by_h_all = abs_err.mean(dim=(0, 2)).numpy()
    mse_by_h_all = sq_err.mean(dim=(0, 2)).numpy()
    rmse_by_h_all = np.sqrt(mse_by_h_all)

    # 昼だけ：各horizonで mask 平均
    # (N,T,1) のまま、hごとに取り出して masked_mean
    T = pred.shape[1]
    mae_by_h_day = np.zeros(T, dtype=np.float32)
    mse_by_h_day = np.zeros(T, dtype=np.float32)
    for t in range(T):
        mae_by_h_day[t] = masked_mean(abs_err[:, t:t+1, :], day[:, t:t+1, :]).item()
        mse_by_h_day[t] = masked_mean(sq_err[:, t:t+1, :], day[:, t:t+1, :]).item()
    rmse_by_h_day = np.sqrt(mse_by_h_day)

    summary = {
        "mae_all": mae_all,
        "mse_all": mse_all,
        "rmse_all": rmse_all,
        "mae_day": mae_day,
        "mse_day": mse_day,
        "rmse_day": rmse_day,
    }
    curves = {
        "mae_by_h_all": mae_by_h_all,
        "mae_by_h_day": mae_by_h_day,
        "rmse_by_h_all": rmse_by_h_all,
        "rmse_by_h_day": rmse_by_h_day,
    }
    return summary, curves


def save_curve_png(y: np.ndarray, title: str, outpath: str, xlabel: str = "horizon (step)", ylabel: str = "error"):
    plt.figure()
    plt.plot(y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def save_timeseries_samples(pred_np: np.ndarray, true_np: np.ndarray, day_np: np.ndarray, outdir: str, n_plot: int = 6):
    """
    サンプルを数個取り、24ステップの予測vs正解を保存。
    day_mask(1)のところは点線などにしたい場合もあるが、
    まずは分かりやすく線2本だけで出す。
    """
    N = pred_np.shape[0]
    T = pred_np.shape[1]
    n_plot = min(int(n_plot), N)

    for i in range(n_plot):
        p = pred_np[i, :, 0]
        y = true_np[i, :, 0]
        d = day_np[i, :, 0]

        plt.figure()
        plt.plot(y, label="true")
        plt.plot(p, label="pred")
        plt.title(f"Timeseries sample {i} (day_ratio={d.mean():.3f})")
        plt.xlabel("time step (0..23)")
        plt.ylabel("PV (same unit as y)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"timeseries_sample{i:02d}.png"))
        plt.close()


def save_scatter_day(pred_np: np.ndarray, true_np: np.ndarray, day_np: np.ndarray, outpath: str, max_points: int = 50000):
    """
    昼だけの (true, pred) 散布図。
    点が多いので最大数を制限。
    """
    p = pred_np.reshape(-1)
    y = true_np.reshape(-1)
    d = day_np.reshape(-1)

    idx = np.where(d > 0.5)[0]
    if idx.size == 0:
        print("[WARN] no day points for scatter.")
        return

    if idx.size > max_points:
        idx = np.random.choice(idx, size=max_points, replace=False)

    plt.figure()
    plt.scatter(y[idx], p[idx], s=2)
    plt.title("Scatter (Day points only): true vs pred")
    plt.xlabel("true PV")
    plt.ylabel("pred PV")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="config .py path")
    parser.add_argument("--ckpt", type=str, required=True, help="checkpoint path (.ckpt)")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"])
    parser.add_argument("--outdir", type=str, default="error_plots")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_plot", type=int, default=6)
    args = parser.parse_args()

    # config 読み込み
    spec = importlib.util.spec_from_file_location("config", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    exp_conf = config_module.exp_conf

    os.makedirs(args.outdir, exist_ok=True)

    # 推論
    pred_np, true_np, day_np = run_inference(
        exp_conf=exp_conf,
        ckpt_path=args.ckpt,
        split=args.split,
        device=args.device,
    )

    # 配列保存（あとで自由に解析できるように）
    np.save(os.path.join(args.outdir, "pred.npy"), pred_np)
    np.save(os.path.join(args.outdir, "true.npy"), true_np)
    np.save(os.path.join(args.outdir, "day_mask.npy"), day_np)

    # 指標計算
    summary, curves = compute_metrics(pred_np, true_np, day_np)

    # サマリ表示（単位は y と同じ）
    # MAE [PV], MSE [PV^2], RMSE [PV]
    print("\n=== Recomputed metrics ===")
    print(f"MAE(all)  = {summary['mae_all']:.6f} [PV]")
    print(f"RMSE(all) = {summary['rmse_all']:.6f} [PV]")
    print(f"MSE(all)  = {summary['mse_all']:.6f} [PV^2]")
    print(f"MAE(day)  = {summary['mae_day']:.6f} [PV]")
    print(f"RMSE(day) = {summary['rmse_day']:.6f} [PV]")
    print(f"MSE(day)  = {summary['mse_day']:.6f} [PV^2]")

    # グラフ保存（horizon別）
    save_curve_png(
        curves["mae_by_h_all"],
        title="MAE by horizon (ALL points)",
        outpath=os.path.join(args.outdir, "mae_by_horizon_all.png"),
        ylabel="MAE [PV]",
    )
    save_curve_png(
        curves["mae_by_h_day"],
        title="MAE by horizon (DAY only)",
        outpath=os.path.join(args.outdir, "mae_by_horizon_day.png"),
        ylabel="MAE [PV]",
    )
    save_curve_png(
        curves["rmse_by_h_all"],
        title="RMSE by horizon (ALL points)",
        outpath=os.path.join(args.outdir, "rmse_by_horizon_all.png"),
        ylabel="RMSE [PV]",
    )
    save_curve_png(
        curves["rmse_by_h_day"],
        title="RMSE by horizon (DAY only)",
        outpath=os.path.join(args.outdir, "rmse_by_horizon_day.png"),
        ylabel="RMSE [PV]",
    )

    # 予測 vs 正解（時系列）
    save_timeseries_samples(pred_np, true_np, day_np, outdir=args.outdir, n_plot=args.n_plot)

    # 散布図（昼だけ）
    save_scatter_day(pred_np, true_np, day_np, outpath=os.path.join(args.outdir, "scatter_day.png"))

    print(f"\n[INFO] saved plots & arrays to: {args.outdir}")


if __name__ == "__main__":
    main()
