# tools/fit_taylor_by_timeband_ghi1h.py
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import importlib.util
import importlib
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import torch


# -----------------------------
# Utility: load config module (exp_conf)
# -----------------------------
def load_exp_conf(config_path: str) -> dict:
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config not found: {config_path}")

    spec = importlib.util.spec_from_file_location("exp_conf_module", config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    if not hasattr(module, "exp_conf"):
        raise KeyError("config module has no exp_conf dict")
    return module.exp_conf


# -----------------------------
# Utility: load model from ckpt
# -----------------------------
def load_model(exp_conf: dict, ckpt_path: str, device: str):
    """
    Try to load a LightningModule-like model.
    - First: class.load_from_checkpoint(ckpt_path, **hparams)
    - Second: instantiate class(**hparams) then load_state_dict
    """
    ckpt_path = os.path.abspath(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    model_module = exp_conf.get("model_module", None)
    model_class_name = exp_conf.get("model_class", None)
    if model_module is None or model_class_name is None:
        raise KeyError("exp_conf must include model_module and model_class")

    mod = importlib.import_module(model_module)
    if not hasattr(mod, model_class_name):
        raise AttributeError(f"{model_module} has no class {model_class_name}")
    cls = getattr(mod, model_class_name)

    # Try 1) load_from_checkpoint (PyTorch Lightning style)
    if hasattr(cls, "load_from_checkpoint"):
        try:
            model = cls.load_from_checkpoint(ckpt_path, **exp_conf)
            model.eval()
            model.to(device)
            return model
        except TypeError:
            # Some classes don't accept **exp_conf; fallback to minimal
            try:
                model = cls.load_from_checkpoint(ckpt_path)
                model.eval()
                model.to(device)
                return model
            except Exception as e:
                print("[WARN] load_from_checkpoint failed:", repr(e))

    # Try 2) instantiate then load_state_dict
    try:
        model = cls(**exp_conf)
    except TypeError:
        # exp_conf has extra keys not in __init__
        # Filter keys by inspecting signature
        import inspect
        sig = inspect.signature(cls.__init__)
        allowed = set(sig.parameters.keys())
        allowed.discard("self")
        filtered = {k: v for k, v in exp_conf.items() if k in allowed}
        model = cls(**filtered)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    # Lightning ckpt often stores "state_dict"
    state_dict = ckpt.get("state_dict", ckpt)

    # If state_dict keys have "model." prefix etc, try to align
    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError:
        # Try stripping common prefixes
        new_sd = {}
        for k, v in state_dict.items():
            nk = re.sub(r"^(model\.)", "", k)
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)

    model.eval()
    model.to(device)
    return model


# -----------------------------
# Time handling (test_t0.npy)
# -----------------------------
def to_jst_datetime64(t0_array: np.ndarray, assume_utc: bool) -> np.ndarray:
    """
    Convert np.datetime64 or integer timestamps to JST np.datetime64[ns].
    If assume_utc=True, add 9 hours.
    """
    a = t0_array

    # Case 1: datetime64
    if np.issubdtype(a.dtype, np.datetime64):
        dt = a.astype("datetime64[ns]")
    else:
        # Case 2: numeric timestamps
        # Heuristic: if values ~1e9 -> seconds, ~1e12 -> ms, ~1e15 -> us, ~1e18 -> ns
        v = a.astype(np.int64)
        absmax = int(np.max(np.abs(v)))
        if absmax < 10**11:
            dt = v.astype("datetime64[s]").astype("datetime64[ns]")
        elif absmax < 10**14:
            dt = v.astype("datetime64[ms]").astype("datetime64[ns]")
        elif absmax < 10**17:
            dt = v.astype("datetime64[us]").astype("datetime64[ns]")
        else:
            dt = v.astype("datetime64[ns]")

    if assume_utc:
        dt = dt + np.timedelta64(9, "h")  # JST = UTC+9
    return dt


def pick_indices_by_hour(t0_jst: np.ndarray, hour_ranges: List[Tuple[int, int]], max_per_band: int) -> List[int]:
    """
    hour_ranges: [(start_hour_inclusive, end_hour_exclusive), ...]
    Returns up to max_per_band indices whose start time hour falls in the ranges.
    """
    # extract hour: convert to "datetime64[h]" then compute hour of day
    # Trick: hour_of_day = (dt - dt.astype('datetime64[D]')) / 1h
    dtD = t0_jst.astype("datetime64[D]")
    hour_of_day = ((t0_jst - dtD) / np.timedelta64(1, "h")).astype(np.int32)

    mask = np.zeros_like(hour_of_day, dtype=bool)
    for h0, h1 in hour_ranges:
        mask |= (hour_of_day >= h0) & (hour_of_day < h1)

    idx = np.where(mask)[0].tolist()
    return idx[:max_per_band]


# -----------------------------
# Probe + Quadratic fit
# -----------------------------
@dataclass
class FitResult:
    a0: float
    a1: float
    a2: float
    r2: float


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def predict_scalar(model, x_window: np.ndarray, device: str, reduce_mode: str) -> float:
    """
    x_window: (T=96, F=12)
    Return scalar from model output for stable curve fitting.
    reduce_mode:
      - "mean96": mean over 96h prediction
      - "daymean": mean only where is_daylight=1 over prediction horizon if available
    """
    xt = torch.from_numpy(x_window[None, ...]).float().to(device)  # (1,96,12)

    with torch.no_grad():
        y = model(xt)

    # Try to normalize output shape to (96,)
    if isinstance(y, (tuple, list)):
        y = y[0]
    y = y.detach().float().cpu().numpy()

    # typical shapes: (1,96), (1,96,1)
    y = np.squeeze(y)
    if y.ndim == 0:
        # model returned a scalar (unlikely)
        return float(y)
    if y.ndim == 2:
        # (96,1) or (1,96) after squeeze; handle
        if y.shape[-1] == 1:
            y = y[:, 0]
        elif y.shape[0] == 1:
            y = y[0, :]
    y = y.reshape(-1)  # (96,)

    if reduce_mode == "mean96":
        return float(np.mean(y))

    if reduce_mode == "daymean":
        # Use is_daylight from INPUT window's last available (or current window) as proxy.
        # In your dataset, is_daylight is feature index 6 (meta.json).
        # This is a practical approximation for slide-ready analysis.
        is_day = x_window[:, 6].astype(np.float32)  # (96,)
        # If it's mostly 0/1, ok. If not, threshold.
        mask = is_day > 0.5
        if np.any(mask):
            return float(np.mean(y[mask]))
        else:
            return float(np.mean(y))

    raise ValueError(f"unknown reduce_mode: {reduce_mode}")


def fit_quadratic(xs: np.ndarray, ys: np.ndarray) -> FitResult:
    # polyfit returns [a2,a1,a0]
    a2, a1, a0 = np.polyfit(xs, ys, 2)
    yhat = a0 + a1 * xs + a2 * (xs ** 2)
    r2 = r2_score(ys, yhat)
    return FitResult(a0=float(a0), a1=float(a1), a2=float(a2), r2=float(r2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_x", required=True)
    ap.add_argument("--test_t0", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--outdir", required=True)

    # Which feature to probe
    ap.add_argument("--probe_feature", default="lag_ghi_wm2_1h",
                    help="feature name in meta.json columns.feature_cols")
    ap.add_argument("--grid_min", type=float, default=0.0)
    ap.add_argument("--grid_max", type=float, default=1000.0)
    ap.add_argument("--grid_step", type=float, default=25.0)

    # Time bands (JST)
    ap.add_argument("--dawn", default="5-7", help="hour range start-end (JST)")
    ap.add_argument("--noon", default="11-13", help="hour range start-end (JST)")
    ap.add_argument("--dusk", default="16-18", help="hour range start-end (JST)")
    ap.add_argument("--max_windows", type=int, default=50, help="max windows per band")

    # t0 timezone assumption
    ap.add_argument("--t0_assume_utc", action="store_true",
                    help="treat test_t0 as UTC and convert to JST by +9h")

    # Reduction for output scalar
    ap.add_argument("--reduce", default="mean96", choices=["mean96", "daymean"])

    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load meta
    with open(args.meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feat_cols = meta["columns"]["feature_cols"]
    if args.probe_feature not in feat_cols:
        raise KeyError(f"probe_feature '{args.probe_feature}' not in feature_cols: {feat_cols}")
    fidx = feat_cols.index(args.probe_feature)

    # Load arrays
    X = np.load(args.test_x)     # (N,96,12)
    t0 = np.load(args.test_t0)   # (N,)
    if X.ndim != 3:
        raise ValueError(f"test_x must be 3D (N,96,F), got {X.shape}")
    N, T, F = X.shape

    # Load model
    exp_conf = load_exp_conf(args.config)
    model = load_model(exp_conf, args.ckpt, args.device)

    # Parse hour ranges
    def parse_range(s: str) -> List[Tuple[int, int]]:
        a, b = s.split("-")
        return [(int(a), int(b))]

    dawn_range = parse_range(args.dawn)
    noon_range = parse_range(args.noon)
    dusk_range = parse_range(args.dusk)

    # Convert t0 to JST and pick indices
    t0_jst = to_jst_datetime64(t0, assume_utc=args.t0_assume_utc)
    dawn_idx = pick_indices_by_hour(t0_jst, dawn_range, args.max_windows)
    noon_idx = pick_indices_by_hour(t0_jst, noon_range, args.max_windows)
    dusk_idx = pick_indices_by_hour(t0_jst, dusk_range, args.max_windows)

    bands = {
        "dawn": dawn_idx,
        "noon": noon_idx,
        "dusk": dusk_idx,
    }

    # Probe grid
    xs = np.arange(args.grid_min, args.grid_max + 1e-9, args.grid_step, dtype=np.float32)

    results: Dict[str, FitResult] = {}

    for band_name, indices in bands.items():
        if len(indices) == 0:
            print(f"[WARN] No windows found for band={band_name}. Check t0 timezone or ranges.")
            continue

        # Use a representative window: take the first one (you can change to median later)
        idx0 = indices[0]
        x_base = X[idx0].copy()  # (96,12)

        ys = []
        for v in xs:
            x_mod = x_base.copy()
            # overwrite PROBED feature across whole 96h history window
            x_mod[:, fidx] = v
            y_scalar = predict_scalar(model, x_mod, device=args.device, reduce_mode=args.reduce)
            ys.append(y_scalar)
        ys = np.array(ys, dtype=np.float32)

        fit = fit_quadratic(xs.astype(np.float64), ys.astype(np.float64))
        results[band_name] = fit

        # Save curve samples for plotting later
        np.savez(
            os.path.join(args.outdir, f"probe_{band_name}.npz"),
            xs=xs, ys=ys, fit=np.array([fit.a0, fit.a1, fit.a2, fit.r2], dtype=np.float64),
            window_index=np.array([idx0], dtype=np.int64),
            window_t0_jst=np.array([str(t0_jst[idx0])]),
            probe_feature=np.array([args.probe_feature]),
            reduce_mode=np.array([args.reduce]),
        )
        print(f"[OK] {band_name}: a0={fit.a0:.6g}, a1={fit.a1:.6g}, a2={fit.a2:.6g}, R2={fit.r2:.4f} (window={idx0})")

    # Save summary CSV + LaTeX
    csv_path = os.path.join(args.outdir, "taylor_fit_summary.csv")
    tex_path = os.path.join(args.outdir, "taylor_fit_summary.tex")

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("band,a0,a1,a2,r2\n")
        for band in ["dawn", "noon", "dusk"]:
            if band in results:
                r = results[band]
                f.write(f"{band},{r.a0},{r.a1},{r.a2},{r.r2}\n")

    # LaTeX equations (slide-ready)
    with open(tex_path, "w", encoding="utf-8") as f:
        for band in ["dawn", "noon", "dusk"]:
            if band in results:
                r = results[band]
                # y(x) = a0 + a1 x + a2 x^2
                f.write(
                    rf"\[\hat{{y}}_{{\mathrm{{{band}}}}}(x)\approx "
                    rf"{r.a0:.4g} + ({r.a1:.4g})x + ({r.a2:.4g})x^2,\quad R^2={r.r2:.3f}\]"
                    "\n"
                )

    print("[OK] saved:", csv_path)
    print("[OK] saved:", tex_path)


if __name__ == "__main__":
    main()
