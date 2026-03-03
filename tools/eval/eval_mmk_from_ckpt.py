# tools/eval_mmk_from_ckpt.py
# -*- coding: utf-8 -*-
"""
ckpt + config + val_x/val_y(+val_marker_y) から、評価指標を計算するツール。

出力:
- prints: MAE / MSE / RMSE / nRMSE / MAPE(非ゼロ) / eps-MAPE
         + しきい値付きMAPE(例: y>=1.0) / Daytime版（マスクがあれば）
- saves:  (outdir)/metrics_eval.json
- saves:  (outdir)/val_pred.npy  (予測値; PV 1変数のみ)
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import torch
except Exception as e:
    raise RuntimeError("PyTorch が import できません。venv を確認してください。") from e


# -----------------------------
# Utils
# -----------------------------
def load_config_module(config_path: str) -> Any:
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config が見つかりません: {config_path}")

    spec = importlib.util.spec_from_file_location("user_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"config の import に失敗: {config_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    if not hasattr(mod, "exp_conf"):
        raise KeyError("config に exp_conf が見つかりません。")
    return mod


def find_existing_file(data_dir: str, candidates: list[str]) -> str:
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "必要な .npy が見つかりません。探した候補:\n"
        + "\n".join([f"- {os.path.join(data_dir, c)}" for c in candidates])
    )


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 0.0) -> float:
    denom = np.abs(y_true) + float(eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_true - y_pred
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


def load_mask(data_dir: str) -> Optional[np.ndarray]:
    candidates = [
        "val_marker_y.npy",
        "val_day_mask_y.npy",
        "val_daymask_y.npy",
        "val_mask_y.npy",
        "val_marker.npy",
        "val_day_mask.npy",
    ]
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return np.load(p)
    return None


def normalize_mask(mask: np.ndarray, y_shape: Tuple[int, ...]) -> np.ndarray:
    if mask.ndim in (2, 3):
        return mask
    if mask.ndim == 1:
        N = y_shape[0]
        pred_len = y_shape[1]
        if mask.shape[0] != N:
            raise ValueError(f"mask shape が不正: mask={mask.shape}, y={y_shape}")
        return np.repeat(mask[:, None], pred_len, axis=1)
    raise ValueError(f"mask ndim が想定外: {mask.ndim}")


def filter_kwargs_for_init(model_cls: Any, exp_conf: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(model_cls.__init__)
    params = sig.parameters

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(exp_conf)

    allowed = {name for name in params.keys() if name != "self"}
    return {k: v for k, v in exp_conf.items() if k in allowed}


# -----------------------------
# Model load
# -----------------------------
def build_model_from_ckpt(exp_conf: Dict[str, Any], ckpt_path: str) -> torch.nn.Module:
    module_name = exp_conf.get("model_module")
    class_name = exp_conf.get("model_class")
    if not module_name or not class_name:
        raise KeyError("exp_conf に model_module / model_class がありません。")

    mod = importlib.import_module(module_name)
    model_cls = getattr(mod, class_name)

    if hasattr(model_cls, "load_from_checkpoint"):
        try:
            return model_cls.load_from_checkpoint(ckpt_path, strict=False)
        except Exception:
            try:
                filtered = filter_kwargs_for_init(model_cls, exp_conf)
                return model_cls.load_from_checkpoint(ckpt_path, strict=False, **filtered)
            except Exception:
                pass

    filtered = filter_kwargs_for_init(model_cls, exp_conf)
    model = model_cls(**filtered)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ["model.", "net.", "module."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        new_sd[nk] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if unexpected:
        print("[WARN] unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")
    if missing:
        print("[WARN] missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    return model


# -----------------------------
# Inference
# -----------------------------
@torch.no_grad()
def predict_val(model: torch.nn.Module, x: np.ndarray, batch_size: int, device: str) -> np.ndarray:
    model.eval()
    model.to(device)

    N = x.shape[0]
    preds = []
    for s in range(0, N, batch_size):
        xb = torch.from_numpy(x[s : s + batch_size]).float().to(device)
        try:
            yb = model(xb)
        except TypeError:
            yb = model.forward(xb)

        if isinstance(yb, (tuple, list)):
            yb = yb[0]
        preds.append(to_numpy(yb))

    return np.concatenate(preds, axis=0)


def compute_mape_with_threshold(y_true: np.ndarray, y_pred: np.ndarray, min_y: float) -> float:
    """|y| >= min_y の要素だけでMAPE[%]を計算（薄明を除外する用途）"""
    mask = np.abs(y_true) >= float(min_y)
    if not np.any(mask):
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=256)

    # MAPE安定化用
    ap.add_argument("--eps_mape", type=float, default=1.0, help="eps-MAPE の eps（発電量単位）")
    ap.add_argument("--mape_min_y", type=float, default=1.0, help="しきい値付きMAPEで使う最小|y|（発電量単位）")

    # 正規化RMSE用
    ap.add_argument("--nrmse_denom", choices=["max", "mean"], default="max", help="nRMSEの分母（max or mean）")

    ap.add_argument("--outdir", default="results_eval_mmk")
    args = ap.parse_args()

    cfg = load_config_module(args.config)
    exp_conf: Dict[str, Any] = dict(cfg.exp_conf)

    data_dir = exp_conf.get("data_dir")
    if not data_dir:
        raise KeyError("exp_conf['data_dir'] がありません。")

    x_path = find_existing_file(data_dir, ["val_x.npy", "val_X.npy", "val_in.npy", "val_input.npy"])
    y_path = find_existing_file(data_dir, ["val_y.npy", "val_Y.npy", "val_out.npy", "val_target.npy"])

    x = np.load(x_path)
    y = np.load(y_path)

    pv_index = int(exp_conf.get("pv_index", 0))

    if y.ndim == 3:
        y_true = y[:, :, pv_index]
    elif y.ndim == 2:
        y_true = y
    else:
        raise ValueError(f"val_y shape が想定外: {y.shape}")

    model = build_model_from_ckpt(exp_conf, args.ckpt)
    y_pred_raw = predict_val(model, x, batch_size=args.batch_size, device=args.device)

    if y_pred_raw.ndim == 3:
        y_pred = y_pred_raw[:, :, 0] if y_pred_raw.shape[2] == 1 else y_pred_raw[:, :, pv_index]
    elif y_pred_raw.ndim == 2:
        y_pred = y_pred_raw
    else:
        raise ValueError(f"pred shape が想定外: {y_pred_raw.shape}")

    metrics = basic_metrics(y_true, y_pred)

    # nRMSE
    denom_val = float(np.max(np.abs(y_true))) if args.nrmse_denom == "max" else float(np.mean(np.abs(y_true)))
    metrics[f"nRMSE_{args.nrmse_denom}[%]"] = float(metrics["RMSE"] / denom_val * 100.0) if denom_val > 0 else float("nan")

    # MAPE 非ゼロ（y=0除外）
    nonzero = np.abs(y_true) > 0.0
    metrics["MAPE_nonzero[%]"] = float(
        np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100.0
    ) if np.any(nonzero) else float("nan")

    # eps-MAPE（推奨：eps=1.0 など）
    metrics[f"MAPE_eps={args.eps_mape:g}[%]"] = safe_mape(y_true, y_pred, eps=args.eps_mape)

    # しきい値付きMAPE（薄明を除外）
    metrics[f"MAPE_|y|>={args.mape_min_y:g}[%]"] = compute_mape_with_threshold(y_true, y_pred, min_y=args.mape_min_y)

    # Daytime（マスクがあれば）
    mask = load_mask(data_dir)
    if mask is not None:
        mask = normalize_mask(mask, y.shape)
        mask2 = mask[:, :, pv_index] if mask.ndim == 3 else mask
        day = mask2 > 0.5

        if np.any(day):
            md = basic_metrics(y_true[day], y_pred[day])
            for k, v in md.items():
                metrics[f"Day_{k}"] = v

            denom_day = float(np.max(np.abs(y_true[day]))) if args.nrmse_denom == "max" else float(np.mean(np.abs(y_true[day])))
            metrics[f"Day_nRMSE_{args.nrmse_denom}[%]"] = float(metrics["Day_RMSE"] / denom_day * 100.0) if denom_day > 0 else float("nan")

            nonzero_day = day & (np.abs(y_true) > 0.0)
            metrics["Day_MAPE_nonzero[%]"] = float(
                np.mean(np.abs((y_true[nonzero_day] - y_pred[nonzero_day]) / y_true[nonzero_day])) * 100.0
            ) if np.any(nonzero_day) else float("nan")

            metrics[f"Day_MAPE_eps={args.eps_mape:g}[%]"] = safe_mape(y_true[day], y_pred[day], eps=args.eps_mape)
            metrics[f"Day_MAPE_|y|>={args.mape_min_y:g}[%]"] = compute_mape_with_threshold(y_true[day], y_pred[day], min_y=args.mape_min_y)
        else:
            metrics["Day_MASK_NOTE"] = "day mask は存在したが day 要素が 0 でした。"
    else:
        metrics["Day_MASK_NOTE"] = "day mask が見つからなかったので Daytime 指標は未計算。"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    np.save(outdir / "val_pred.npy", y_pred.astype(np.float32))

    with open(outdir / "metrics_eval.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": os.path.abspath(args.config),
                "ckpt": os.path.abspath(args.ckpt),
                "data_dir": data_dir,
                "x_path": x_path,
                "y_path": y_path,
                "pv_index": pv_index,
                "eps_mape": args.eps_mape,
                "mape_min_y": args.mape_min_y,
                "nrmse_denom": args.nrmse_denom,
                "metrics": metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n======== EVAL RESULTS (PV only) ========")
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, float):
            print(f"{k:28s}: {v:.6g}")
        else:
            print(f"{k:28s}: {v}")
    print(f"\n[Saved] {outdir / 'metrics_eval.json'}")
    print(f"[Saved] {outdir / 'val_pred.npy'}")
    print("=======================================\n")


if __name__ == "__main__":
    main()
