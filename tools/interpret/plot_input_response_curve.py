# tools/plot_input_response_curve.py
# ------------------------------------------------------------
# Input-Output Response (what-if) analysis for MMK_FusionPV (KAN)
# - No retraining required (inference only)
# - Perturb ONE input feature continuously (e.g., ghi_wm2 scale)
# - Plot how a scalar summary of the 96h forecast changes
#
# Outputs:
#   outdir/response_curve.png
#   outdir/response_curve.csv
#   (optional) outdir/debug_pred_example.png
# ------------------------------------------------------------

import os
import json
import argparse
import importlib.util
import importlib
import inspect
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


# -----------------------------
# Utility: load exp_conf from config python file
# -----------------------------
def load_exp_conf(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config not found: {config_path}")
    spec = importlib.util.spec_from_file_location("exp_config_module", config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    if not hasattr(module, "exp_conf"):
        raise ValueError(f"exp_conf not found in config: {config_path}")
    return module.exp_conf


# -----------------------------
# Utility: robust feature name list loader for your meta.json
# -----------------------------
def load_feature_names(meta_json_path: str) -> List[str]:
    if not os.path.exists(meta_json_path):
        raise FileNotFoundError(f"meta_json not found: {meta_json_path}")
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Case A: directly stored list
    candidates_top = ["feature_names", "x_cols", "columns", "input_features", "feat_names"]
    for k in candidates_top:
        if k in meta and isinstance(meta[k], list) and len(meta[k]) > 0:
            return meta[k]

    # Case B: nested under meta["data"]
    if isinstance(meta.get("data", None), dict):
        for k in candidates_top:
            v = meta["data"].get(k, None)
            if isinstance(v, list) and len(v) > 0:
                return v

    # Case C (your format): meta["columns"] is dict with "feature_cols"
    cols = meta.get("columns", None)
    if isinstance(cols, dict):
        for k in ["feature_cols", "features", "x_cols", "feature_names", "input_features"]:
            v = cols.get(k, None)
            if isinstance(v, list) and len(v) > 0:
                return v

    top_keys = list(meta.keys())
    raise ValueError(
        "Could not find feature name list in meta.json.\n"
        "Tried keys: feature_names / x_cols / columns(list) / input_features,\n"
        "and columns.feature_cols (dict-style).\n"
        f"Top-level keys in your meta.json: {top_keys}\n"
        "Tip: open meta.json and locate the list of feature column names."
    )


def get_feature_index(feature_names: List[str], target_name: str) -> int:
    if target_name in feature_names:
        return feature_names.index(target_name)

    low = [str(x).lower() for x in feature_names]
    t = target_name.lower()
    if t in low:
        return low.index(t)

    for i, name in enumerate(low):
        if t in name:
            return i

    raise ValueError(
        f"Feature '{target_name}' not found.\n"
        f"Available features ({len(feature_names)}): {feature_names}"
    )


# -----------------------------
# Model loading (重要: traingraphレジストリ依存を回避)
# - まず traingraph.get_model を試す
# - ダメなら exp_conf["model_module"] を import して class を直接生成
# -----------------------------
def _instantiate_from_module(model_module: str, model_class_name: str, exp_conf: Dict[str, Any]) -> torch.nn.Module:
    """
    model_module: e.g. "easytsf.model.MMK_FusionPV_FeatureToken_v2"
    model_class_name: e.g. "MMK_FusionPV_FeatureToken"
    exp_conf: config dict (we will filter keys by __init__ signature)
    """
    mod = importlib.import_module(model_module)
    if not hasattr(mod, model_class_name):
        raise AttributeError(
            f"Class '{model_class_name}' not found in module '{model_module}'. "
            f"Available attrs example: {[a for a in dir(mod) if 'MMK' in a or 'Fusion' in a][:20]}"
        )
    cls = getattr(mod, model_class_name)

    sig = inspect.signature(cls.__init__)
    valid_keys = set(sig.parameters.keys())
    valid_keys.discard("self")

    # exp_conf contains many non-init keys; filter them
    kwargs = {k: v for k, v in exp_conf.items() if k in valid_keys}

    # NOTE: if your model expects nested dicts, they are already in exp_conf (e.g., mok_conf)
    model = cls(**kwargs)
    return model


def build_model_from_exp_conf(exp_conf: Dict[str, Any], ckpt_path: str, device: str) -> torch.nn.Module:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    model_name = exp_conf.get("model_name", None)
    if model_name is None:
        raise ValueError("exp_conf['model_name'] not found in config")

    # 1) Try traingraph.get_model (if registry supports it)
    model = None
    traingraph_error = None
    try:
        from traingraph import get_model  # type: ignore
        model = get_model(model_name, exp_conf)
        print("[INFO] model built via traingraph.get_model")
    except Exception as e:
        traingraph_error = e
        model = None

    # 2) Fallback: import exp_conf["model_module"] and instantiate class directly
    if model is None:
        model_module = exp_conf.get("model_module", None)
        if model_module is None:
            raise RuntimeError(
                "Model not supported by traingraph.get_model and exp_conf['model_module'] is missing.\n"
                f"traingraph error: {traingraph_error}"
            )
        print(f"[INFO] traingraph.get_model failed ({traingraph_error}). Fallback to model_module import: {model_module}")
        model = _instantiate_from_module(model_module, model_name, exp_conf)

    # Load ckpt (Lightning ckpt typically has 'state_dict')
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    def strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                out[k[len(prefix):]] = v
            else:
                out[k] = v
        return out

    load_ok = False
    last_msg = None
    for prefix in ["model.", "net.", "module.", ""]:
        sd2 = strip_prefix(state_dict, prefix) if prefix else state_dict
        missing, unexpected = model.load_state_dict(sd2, strict=False)
        # "全部missing" に近い場合は失敗扱い（モデルが違う可能性）
        if len(missing) < len(sd2):
            load_ok = True
            print(f"[INFO] loaded state_dict with prefix='{prefix}'  missing={len(missing)} unexpected={len(unexpected)}")
            break
        last_msg = (missing, unexpected)

    if not load_ok:
        raise RuntimeError(
            "Failed to load checkpoint weights into model. "
            "Config/model_name may not match checkpoint.\n"
            f"Last load attempt missing/unexpected: {last_msg}"
        )

    model.eval()
    model.to(device)
    return model


# -----------------------------
# Forward helper (robust)
# -----------------------------
@torch.no_grad()
def predict_y(model: torch.nn.Module, x_win: torch.Tensor) -> torch.Tensor:
    """
    x_win: (1, T_in, F)
    returns y_pred: (1, T_out) or (1, T_out, 1) or (1, T_out, C)
    """
    # A) direct forward
    try:
        y = model(x_win)
        if isinstance(y, (tuple, list)):
            y = y[0]
        return y
    except Exception:
        pass

    # B) dict-style
    try:
        y = model({"x": x_win})
        if isinstance(y, (tuple, list)):
            y = y[0]
        return y
    except Exception:
        pass

    # C) Lightning-like predict_step
    if hasattr(model, "predict_step"):
        try:
            y = model.predict_step({"x": x_win}, 0)
            if isinstance(y, (tuple, list)):
                y = y[0]
            return y
        except Exception:
            pass

    raise RuntimeError(
        "Could not run model forward. "
        "Paste the error from calling model(x) so I can adapt the wrapper."
    )


# -----------------------------
# Summarize 96h output into 1 scalar for response curve
# -----------------------------
def summarize_pred(y_pred: np.ndarray, mode: str, hour_index: int) -> float:
    """
    y_pred: (T_out,) or (T_out, C)
    mode:
      - peak: max over time
      - energy: sum over time (assumes 1h step; unit is 'pred_unit * h')
      - hour: value at hour_index
    """
    if y_pred.ndim == 2:
        y_pred_1d = y_pred[:, 0]
    else:
        y_pred_1d = y_pred

    if mode == "peak":
        return float(np.max(y_pred_1d))
    if mode == "energy":
        return float(np.sum(y_pred_1d))
    if mode == "hour":
        if hour_index < 0 or hour_index >= y_pred_1d.shape[0]:
            raise ValueError(f"hour_index out of range: {hour_index} for T_out={y_pred_1d.shape[0]}")
        return float(y_pred_1d[hour_index])

    raise ValueError(f"Unknown summary mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="path to config .py containing exp_conf")
    ap.add_argument("--ckpt", required=True, help="path to trained checkpoint .ckpt")
    ap.add_argument("--meta_json", required=True, help="path to meta.json (feature names)")
    ap.add_argument("--test_x", required=True, help="path to test_x.npy (N, T_in, F)")
    ap.add_argument("--outdir", default="results_response", help="output directory")

    ap.add_argument("--feature", default="ghi_wm2", help="feature name to perturb (must exist in meta.json)")
    ap.add_argument("--window_index", type=int, default=0, help="which window index from test_x.npy to use")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="device")

    ap.add_argument("--s_min", type=float, default=0.7, help="min scale")
    ap.add_argument("--s_max", type=float, default=1.3, help="max scale")
    ap.add_argument("--s_steps", type=int, default=13, help="number of scales")

    ap.add_argument("--summary", default="peak", choices=["peak", "energy", "hour"],
                    help="scalar summary of 96h output")
    ap.add_argument("--hour_index", type=int, default=12,
                    help="used only if --summary hour (0..95)")

    ap.add_argument("--debug_scale", type=float, default=1.0, help="scale to plot full 96h pred (set to -1 to disable)")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    feat_names = load_feature_names(args.meta_json)
    fidx = get_feature_index(feat_names, args.feature)
    print(f"[INFO] feature='{args.feature}' index={fidx} / F={len(feat_names)}")
    print(f"[INFO] features: {feat_names}")

    x = np.load(args.test_x)  # (N, T_in, F)
    if x.ndim != 3:
        raise ValueError(f"test_x must be (N, T_in, F). got shape={x.shape}")
    if args.window_index < 0 or args.window_index >= x.shape[0]:
        raise ValueError(f"window_index out of range: {args.window_index} for N={x.shape[0]}")

    x_win0 = x[args.window_index].copy()  # (T_in, F)

    exp_conf = load_exp_conf(args.config)
    model = build_model_from_exp_conf(exp_conf, args.ckpt, device=args.device)

    scales = np.linspace(args.s_min, args.s_max, args.s_steps).astype(np.float32)
    rows: List[Tuple[float, float]] = []

    for s in scales:
        x_win = x_win0.copy()
        x_win[:, fidx] = x_win[:, fidx] * float(s)

        x_t = torch.from_numpy(x_win).float().unsqueeze(0).to(args.device)  # (1, T_in, F)
        y_t = predict_y(model, x_t)

        y_np = y_t.detach().cpu().numpy()
        y_np = np.squeeze(y_np, axis=0)

        score = summarize_pred(y_np, mode=args.summary, hour_index=args.hour_index)
        rows.append((float(s), float(score)))
        print(f"[S={s:.3f}] score({args.summary}) = {score:.6f}")

    csv_path = os.path.join(args.outdir, "response_curve.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("scale,score\n")
        for s, sc in rows:
            f.write(f"{s},{sc}\n")
    print(f"[OK] saved: {csv_path}")

    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("input scale (s)")
    plt.ylabel(f"output score ({args.summary})")
    plt.title(f"Input-Output Response: feature='{args.feature}' window={args.window_index}")
    plt.grid(True)

    fig_path = os.path.join(args.outdir, "response_curve.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved: {fig_path}")

    if args.debug_scale >= 0:
        s = float(args.debug_scale)
        x_win = x_win0.copy()
        x_win[:, fidx] = x_win[:, fidx] * s
        x_t = torch.from_numpy(x_win).float().unsqueeze(0).to(args.device)
        y_t = predict_y(model, x_t)
        y_np = np.squeeze(y_t.detach().cpu().numpy(), axis=0)

        if y_np.ndim == 2:
            y_np = y_np[:, 0]

        plt.figure()
        plt.plot(np.arange(len(y_np)), y_np)
        plt.xlabel("forecast hour (0..)")
        plt.ylabel("pred")
        plt.title(f"96h prediction at scale={s:.3f} (feature='{args.feature}')")
        plt.grid(True)

        dbg_path = os.path.join(args.outdir, "debug_pred_example.png")
        plt.savefig(dbg_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[OK] saved: {dbg_path}")


if __name__ == "__main__":
    main()
