# -*- coding: utf-8 -*-
"""
plot_pred_vs_true_itr.py（完全版）

目的
- iTransformerPeak のチェックポイントから test データの予測を作成し、
  ① 指定window（デフォルト: 最後）の pred vs true
  ② （可能なら）t0 付きのタイムライン pred vs true
  を保存します。

この版で直っていること
- --window_index -1 を「最後の窓」として正しく扱う
- 0未満の予測値を、評価・可視化時だけ 0 にクリップできる（--clip0）
- test_t0.npy が存在する場合は “時刻軸” の timeline_plot を作る
  （無い場合は stitched_timeline_plot にフォールバック）

使い方例（最後の窓を表示）
python tools\plot_pred_vs_true_itr.py -c config\\tsukuba_conf\\iTransformer_peak_96_noscale.py --ckpt "C:\\...\\best.ckpt" --window_index -1 --outdir results_predplots --clip0
"""

import os
import json
import argparse
import importlib.util
import inspect
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import matplotlib.pyplot as plt


# ----------------------------
# Utility
# ----------------------------
def load_config(config_path: str) -> Dict[str, Any]:
    """config py を import して exp_conf を取り出す"""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"[ERROR] failed to load config: {config_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "exp_conf"):
        raise AttributeError(f"[ERROR] config has no exp_conf: {config_path}")
    return mod.exp_conf


def to_2d(arr: np.ndarray) -> np.ndarray:
    """
    予測/教師の配列形状を (B, H) に正規化する
    - (B, H) -> OK
    - (B, H, 1) -> (B, H)
    - (B, H, C) -> Cがあっても先頭チャネルを使う（事故回避）
    """
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[:, :, 0]
    raise ValueError(f"[ERROR] unexpected array shape: {arr.shape}")


def resolve_window_index(wi: int, n_windows: int) -> int:
    """
    --window_index の解決
    - wi>=0: そのまま
    - wi<0 : Python流（-1=最後、-2=最後から2番目…）
    最後に範囲外を丸める
    """
    if n_windows <= 0:
        raise ValueError("[ERROR] n_windows must be positive.")
    if wi < 0:
        wi = n_windows + wi
    wi = int(np.clip(wi, 0, n_windows - 1))
    return wi


def find_test_arrays(data_dir: str) -> Dict[str, str]:
    """
    data_dir の中から test 用ファイルを探す
    config側でファイル名が変わっても最低限動くようにするための保険
    """
    paths = {}
    for name in ["test_x.npy", "test_y.npy", "test_t0.npy", "test_marker_y.npy"]:
        p = os.path.join(data_dir, name)
        if os.path.isfile(p):
            paths[name] = p
    # 必須
    if "test_x.npy" not in paths or "test_y.npy" not in paths:
        raise FileNotFoundError(
            f"[ERROR] test_x.npy / test_y.npy not found under: {data_dir}"
        )
    return paths


def batched_predict(model: torch.nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    """numpy (B, L, F) をモデルに入れて numpy で返す"""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i:i + batch_size]).float()
            # iTransformerPeak は forward(x, marker_x=None) を想定
            yb = model(xb, None)
            if isinstance(yb, (tuple, list)):
                yb = yb[0]
            outs.append(yb.detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


# ----------------------------
# Model loader (iTransformerPeak)
# ----------------------------
def get_model(model_name: str, exp_conf: Dict[str, Any]) -> torch.nn.Module:
    """
    configの exp_conf から iTransformerPeak を生成する
    __init__ の引数と一致するキーだけを渡す（余計なキーで落ちないように）
    """
    if model_name == "iTransformerPeak":
        from easytsf.model.iTransformer_peak import iTransformerPeak
        model_class = iTransformerPeak
    else:
        raise ValueError(f"[ERROR] unsupported model_name: {model_name}")

    sig = inspect.signature(model_class.__init__)
    valid_params = [p.name for p in sig.parameters.values() if p.name != "self"]
    filtered = {k: v for k, v in exp_conf.items() if k in valid_params}
    return model_class(**filtered)


def load_base_model_from_ckpt(base_model: torch.nn.Module, ckpt_path: str) -> torch.nn.Module:
    """
    Lightning ckpt / state_dict を読み、ベースモデルにロードする
    state_dict が "model.xxx" で入っている場合は prefix を剥がす
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            new_sd[k[len("model."):]] = v

    base_model.load_state_dict(new_sd if new_sd else sd, strict=False)
    return base_model


# ----------------------------
# Timeline builders
# ----------------------------
def build_timeline_from_t0(
    t0: np.ndarray,
    seq: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    t0: (B,) datetime64[ns]
    seq: (B, H)
    1時間刻みで (t0[i] + k[h]) に割り当て、同時刻は平均
    """
    bucket = {}
    B, H = seq.shape
    for i in range(B):
        start = t0[i]
        for k in range(H):
            ts = start + np.timedelta64(k, "h")
            bucket.setdefault(ts, []).append(float(seq[i, k]))

    ts_sorted = np.array(sorted(bucket.keys()), dtype="datetime64[ns]")
    vals = np.array([np.mean(bucket[t]) for t in ts_sorted], dtype=np.float32)
    return ts_sorted, vals


def build_stitched_timeline(seq: np.ndarray) -> np.ndarray:
    """
    t0 がない場合のフォールバック：
    (B, H) を単純に 1 本へ連結
    """
    return seq.reshape(-1)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("-s", "--save_dir", type=str, default="save")  # 互換のため残す（未使用でもOK）
    ap.add_argument("--outdir", type=str, default="results_predplots_itr")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--window_index", type=int, default=-1)  # -1=最後
    ap.add_argument("--max_points", type=int, default=2000)  # timeline用間引き
    ap.add_argument("--clip0", action="store_true", help="clip prediction to >=0 at eval/plot time only")
    args = ap.parse_args()

    exp = load_config(args.config)
    model_name = exp.get("model_name", "iTransformerPeak")
    data_dir = exp.get("data_dir", None)
    if data_dir is None:
        raise KeyError("[ERROR] exp_conf has no 'data_dir'")

    ckpt_path = os.path.normpath(args.ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"[ERROR] ckpt not found: {ckpt_path}")

    os.makedirs(args.outdir, exist_ok=True)

    print("[INFO] model_name:", model_name)
    print("[INFO] data_dir:", data_dir)
    print("[INFO] using ckpt:", ckpt_path)

    # Load data (test)
    found = find_test_arrays(data_dir)
    test_x = np.load(found["test_x.npy"])
    test_y = np.load(found["test_y.npy"])
    test_t0 = np.load(found["test_t0.npy"]) if "test_t0.npy" in found else None

    y_true = to_2d(test_y)

    # Load model + weights
    model = get_model(model_name, exp)
    model = load_base_model_from_ckpt(model, ckpt_path)

    # Predict
    pred_raw = batched_predict(model, test_x, args.batch_size)
    pred = to_2d(pred_raw)

    # Optional: clip negative predictions at eval/plot time only
    if args.clip0:
        pred = np.maximum(pred, 0.0)

    # Resolve window index (supports negative)
    B = y_true.shape[0]
    wi = resolve_window_index(args.window_index, B)
    print(f"[INFO] n_windows={B}, window_index(requested)={args.window_index}, window_index(actual)={wi}")

    # ----------------------------
    # (1) Window plot
    # ----------------------------
    plt.figure()
    plt.plot(y_true[wi], label="true")
    plt.plot(pred[wi], label="pred")
    title = f"{model_name} test window idx={wi}"
    if test_t0 is not None:
        title += f"  start={str(test_t0[wi])}"
    if args.clip0:
        title += "  (clip0)"
    plt.title(title)
    plt.xlabel("Horizon step (hour)")
    plt.ylabel("pv_kwh" if "noscale" in str(data_dir).lower() else "pv (maybe scaled)")
    plt.legend()
    out_window = os.path.join(args.outdir, "window_plot.png")
    plt.savefig(out_window, dpi=150, bbox_inches="tight")
    print("[OK] saved:", out_window)

    # ----------------------------
    # (2) Timeline plot
    # ----------------------------
    if test_t0 is not None:
        # Build timeline with actual timestamps
        ts_true, v_true = build_timeline_from_t0(test_t0, y_true)
        ts_pred, v_pred = build_timeline_from_t0(test_t0, pred)

        # Align on common timestamps
        common = np.intersect1d(ts_true, ts_pred)
        it = np.searchsorted(ts_true, common)
        ip = np.searchsorted(ts_pred, common)
        ts = common
        vt = v_true[it]
        vp = v_pred[ip]

        # Downsample if too long
        if ts.shape[0] > args.max_points:
            step = int(np.ceil(ts.shape[0] / args.max_points))
            ts = ts[::step]
            vt = vt[::step]
            vp = vp[::step]

        plt.figure(figsize=(12, 4))
        plt.plot(ts.astype("datetime64[h]"), vt, label="true")
        plt.plot(ts.astype("datetime64[h]"), vp, label="pred")
        ttitle = f"{model_name} test timeline (overlaps averaged)"
        if args.clip0:
            ttitle += "  (clip0)"
        plt.title(ttitle)
        plt.xlabel("time")
        plt.ylabel("pv_kwh" if "noscale" in str(data_dir).lower() else "pv (maybe scaled)")
        plt.legend()
        out_timeline = os.path.join(args.outdir, "timeline_plot.png")
        plt.savefig(out_timeline, dpi=150, bbox_inches="tight")
        print("[OK] saved:", out_timeline)
    else:
        # Fallback: stitched (no timestamps)
        true_flat = build_stitched_timeline(y_true)
        pred_flat = build_stitched_timeline(pred)

        n = true_flat.shape[0]
        if n > args.max_points:
            step = int(np.ceil(n / args.max_points))
            true_flat = true_flat[::step]
            pred_flat = pred_flat[::step]

        plt.figure(figsize=(12, 4))
        plt.plot(true_flat, label="true")
        plt.plot(pred_flat, label="pred")
        ttitle = f"{model_name} stitched timeline (no test_t0)"
        if args.clip0:
            ttitle += "  (clip0)"
        plt.title(ttitle)
        plt.xlabel("index (stitched)")
        plt.ylabel("pv_kwh" if "noscale" in str(data_dir).lower() else "pv (maybe scaled)")
        plt.legend()
        out_stitched = os.path.join(args.outdir, "stitched_timeline_plot.png")
        plt.savefig(out_stitched, dpi=150, bbox_inches="tight")
        print("[OK] saved:", out_stitched)

    # ----------------------------
    # Save arrays + info
    # ----------------------------
    np.save(os.path.join(args.outdir, "pred_test.npy"), pred.astype(np.float32))
    np.save(os.path.join(args.outdir, "true_test.npy"), y_true.astype(np.float32))

    info = {
        "ckpt": ckpt_path,
        "config": args.config,
        "data_dir": data_dir,
        "model_name": model_name,
        "n_windows": int(B),
        "window_index_requested": int(args.window_index),
        "window_index_actual": int(wi),
        "batch_size": int(args.batch_size),
        "clip0": bool(args.clip0),
        "note": "If test_t0.npy exists, timeline_plot.png is timestamped; otherwise stitched_timeline_plot.png is saved.",
    }
    with open(os.path.join(args.outdir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print("[OK] saved:", os.path.join(args.outdir, "info.json"))


if __name__ == "__main__":
    main()
