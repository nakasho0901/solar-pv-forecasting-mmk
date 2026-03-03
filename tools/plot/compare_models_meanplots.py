# -*- coding: utf-8 -*-
r"""
compare_models_meanplots.py

MMK と iTransformerPeak を「1回の実行」で推論→平均化→比較プロットするスクリプト。
plot_pred_vs_true_revin.py の思想（config/ckpt駆動、同じ前処理・平均化）を踏襲しつつ、
既存スクリプトは変更しない（MMK構造を壊さない）。

出力:
- compare_mean96h_pred_vs_true_raw.png
- compare_mean96h_pred_vs_true_night0.png
- compare_hour_of_day_mean_raw.png         (t0があれば)
- compare_hour_of_day_mean_night0.png      (t0があれば)
- compare_horizon_mean_raw.png
- compare_horizon_mean_night0.png
- （任意）各モデルの予測/真値 npy を outdir 配下に保存

前提（各data_dirにあるもの）:
- {split}_x.npy, {split}_y.npy は必須
- {split}_t0.npy, {split}_marker_y.npy は推奨（hour-of-day / night0に使用）
- {split}_marker_x.npy は任意（あれば iTr に渡す試行をする）
"""

import os
import json
import argparse
import importlib
import importlib.util
from typing import Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt


# ----------------------------
# config
# ----------------------------
def load_config(config_path: str) -> dict:
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.exp_conf


# ----------------------------
# model loader (MMK互換 + iTr)
# ----------------------------
def get_model(exp_conf: dict):
    """
    config の model_module / model_class があれば優先。
    無い場合は model_name で fallback。
    """
    model_module = exp_conf.get("model_module", None)
    model_class_name = exp_conf.get("model_class", None)
    model_name = exp_conf.get("model_name", None)

    if model_module and model_class_name:
        mod = importlib.import_module(model_module)
        model_class = getattr(mod, model_class_name)
    else:
        if model_name == "MMK_FusionPV_FeatureToken":
            from easytsf.model.MMK_FusionPV_FeatureToken import MMK_FusionPV_FeatureToken
            model_class = MMK_FusionPV_FeatureToken
        elif model_name == "iTransformerPeak":
            from easytsf.model.iTransformer_peak import iTransformerPeak
            model_class = iTransformerPeak
        else:
            raise ValueError(f"[ERROR] model resolve failed: model_name={model_name}")

    import inspect
    sig = inspect.signature(model_class.__init__)
    valid_params = [p.name for p in sig.parameters.values() if p.name != "self"]
    filtered = {k: v for k, v in exp_conf.items() if k in valid_params}
    return model_class(**filtered)


def load_base_model_from_ckpt(base_model, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt)

    # Lightning の "model.xxx" を剥がす
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            new_sd[k[len("model."):]] = v

    if new_sd:
        base_model.load_state_dict(new_sd, strict=False)
    else:
        base_model.load_state_dict(sd, strict=False)

    return base_model.eval()


# ----------------------------
# data
# ----------------------------
def load_split_arrays(data_dir: str, split: str):
    split = split.lower()
    x_path = os.path.join(data_dir, f"{split}_x.npy")
    y_path = os.path.join(data_dir, f"{split}_y.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"[ERROR] not found: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"[ERROR] not found: {y_path}")

    x = np.load(x_path)
    y = np.load(y_path)

    t0_path = os.path.join(data_dir, f"{split}_t0.npy")
    t0 = np.load(t0_path) if os.path.exists(t0_path) else None

    marker_y_path = os.path.join(data_dir, f"{split}_marker_y.npy")
    marker_y = np.load(marker_y_path) if os.path.exists(marker_y_path) else None

    marker_x_path = os.path.join(data_dir, f"{split}_marker_x.npy")
    marker_x = np.load(marker_x_path) if os.path.exists(marker_x_path) else None

    return x, y, t0, marker_y, marker_x


def apply_ghi_boost(x: np.ndarray, ghi_index: int, factor: float) -> np.ndarray:
    if factor == 1.0:
        return x
    x2 = x.copy()
    if 0 <= ghi_index < x2.shape[-1]:
        x2[:, :, ghi_index] *= factor
    return x2


# ----------------------------
# inference (MMK互換を壊さない forward 自動判定)
# ----------------------------
def batched_predict(model, x: np.ndarray, batch_size: int = 64, marker_x: Optional[np.ndarray] = None) -> np.ndarray:
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, x.shape[0], batch_size):
            xb = torch.from_numpy(x[i:i + batch_size]).float()

            mb = None
            if marker_x is not None:
                mb = torch.from_numpy(marker_x[i:i + batch_size]).float()

            out = None
            last_err = None

            # 1) marker_xがあるなら渡してみる（iTrなど）
            try:
                out = model(xb, mb)
            except Exception as e:
                last_err = e

            # 2) 従来互換（MMK）
            if out is None:
                try:
                    out = model(xb, None)
                except Exception as e:
                    last_err = e

            # 3) 1引数 forward（iTrなど）
            if out is None:
                try:
                    out = model(xb)
                except Exception as e:
                    last_err = e

            # 4) marker必須モデル用ダミー
            if out is None:
                try:
                    B, L, _ = xb.shape
                    dummy_mark = torch.zeros((B, L, 1), dtype=xb.dtype, device=xb.device)
                    out = model(xb, dummy_mark)
                except Exception as e:
                    last_err = e

            if out is None:
                raise RuntimeError(f"[ERROR] model forward failed. last_err={repr(last_err)}")

            if isinstance(out, (tuple, list)):
                out = out[0]
            preds.append(out.cpu().numpy())
    return np.concatenate(preds, axis=0)


def extract_target(arr: np.ndarray, target_index: int) -> np.ndarray:
    """
    arr: (N, T, C) or (N, T)
    return: (N, T)
    """
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"[ERROR] unexpected array shape: {arr.shape}")
    if arr.shape[-1] == 1:
        return arr[:, :, 0]
    if 0 <= target_index < arr.shape[-1]:
        return arr[:, :, target_index]
    return arr[:, :, 0]


def apply_night0_mask(seq: np.ndarray, marker_y: Optional[np.ndarray]) -> np.ndarray:
    """
    seq: (N, T)
    marker_y: (N, T) or (N, T, 1) or None
    """
    if marker_y is None:
        return seq
    m = marker_y.squeeze(-1) if marker_y.ndim == 3 else marker_y
    return seq * m


# ----------------------------
# JST hour-of-day aggregation
# ----------------------------
def hour_of_day_from_t0(t0: np.ndarray, tz_offset_hours: int) -> np.ndarray:
    hour_utc = t0.astype("datetime64[h]").astype("int64")
    hour_local = (hour_utc + int(tz_offset_hours)) % 24
    return hour_local.astype(np.int32)


def hour_of_day_mean(t0: np.ndarray, y: np.ndarray, pred_len: int, tz_offset_hours: int) -> np.ndarray:
    hod0 = hour_of_day_from_t0(t0, tz_offset_hours)
    N = y.shape[0]
    sumv = np.zeros(24, dtype=np.float64)
    cnt = np.zeros(24, dtype=np.int64)
    for i in range(N):
        base = int(hod0[i])
        for k in range(pred_len):
            h = (base + k) % 24
            sumv[h] += float(y[i, k])
            cnt[h] += 1
    return (sumv / np.maximum(cnt, 1)).astype(np.float32)


# ----------------------------
# plots
# ----------------------------
def plot_mean96h_compare(true_mean, mmk_mean, itr_mean, outpath, title):
    h = np.arange(len(true_mean))
    plt.figure(figsize=(11, 4))
    plt.plot(h, true_mean, label="True mean", linewidth=2)
    plt.plot(h, mmk_mean, label="MMK pred mean", linewidth=2)
    plt.plot(h, itr_mean, label="iTr pred mean", linewidth=2)
    plt.xlabel("Horizon step [hour]")
    plt.ylabel("PV")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


def plot_horizon_mean_compare(true_mean, mmk_mean, itr_mean, outpath, title):
    # 見た目は mean96h と同じ軸だが、タイトルで区別（研究用に明示）
    plot_mean96h_compare(true_mean, mmk_mean, itr_mean, outpath, title)


def plot_hour_of_day_compare(true_hod, mmk_hod, itr_hod, outpath, title):
    h = np.arange(24)
    plt.figure(figsize=(11, 4))
    plt.plot(h, true_hod, label="True mean (JST)", linewidth=2)
    plt.plot(h, mmk_hod, label="MMK pred mean (JST)", linewidth=2)
    plt.plot(h, itr_hod, label="iTr pred mean (JST)", linewidth=2)
    plt.xlabel("Hour of Day [hour] (JST)")
    plt.ylabel("PV")
    plt.title(title)
    plt.xticks(h)
    plt.grid(True)
    plt.legend()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()


# ----------------------------
# main pipeline per model
# ----------------------------
def run_one_model(
    name: str,
    config_path: str,
    ckpt_path: str,
    split: str,
    batch_size: int,
    tz_offset_hours: int,
    save_arrays_dir: Optional[str] = None,
) -> dict:
    exp = load_config(config_path)

    data_dir = exp["data_dir"]
    pv_index = int(exp.get("pv_index", 0))
    ghi_index = int(exp.get("ghi_index", 3))
    ghi_boost = float(exp.get("ghi_boost", 1.0))
    pred_len = int(exp.get("pred_len", exp.get("fut_len", 96)))

    x, y, t0, marker_y, marker_x = load_split_arrays(data_dir, split)

    model = get_model(exp)
    model = load_base_model_from_ckpt(model, ckpt_path)

    x_in = apply_ghi_boost(x, ghi_index=ghi_index, factor=ghi_boost)
    pred_all = batched_predict(model, x_in, batch_size=batch_size, marker_x=marker_x)

    pred_pv_raw = extract_target(pred_all, pv_index)     # (N, T)
    true_pv = extract_target(y, pv_index)                # (N, T)

    pred_pv_night0 = apply_night0_mask(pred_pv_raw, marker_y)
    true_pv_night0 = apply_night0_mask(true_pv, marker_y)

    out = {
        "name": name,
        "config": config_path,
        "ckpt": ckpt_path,
        "data_dir": data_dir,
        "pred_len": pred_len,
        "has_t0": t0 is not None,
        "has_marker_y": marker_y is not None,
        "pred_raw": pred_pv_raw,
        "pred_night0": pred_pv_night0,
        "true_raw": true_pv,
        "true_night0": true_pv_night0,
        "t0": t0,
    }

    if save_arrays_dir is not None:
        os.makedirs(save_arrays_dir, exist_ok=True)
        np.save(os.path.join(save_arrays_dir, f"{name}_pred_raw.npy"), pred_pv_raw.astype(np.float32))
        np.save(os.path.join(save_arrays_dir, f"{name}_pred_night0.npy"), pred_pv_night0.astype(np.float32))
        np.save(os.path.join(save_arrays_dir, f"{name}_true.npy"), true_pv.astype(np.float32))
        if marker_y is not None:
            np.save(os.path.join(save_arrays_dir, f"{name}_marker_y.npy"), marker_y)
        if t0 is not None:
            np.save(os.path.join(save_arrays_dir, f"{name}_t0.npy"), t0)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmk_config", required=True)
    ap.add_argument("--mmk_ckpt", required=True)
    ap.add_argument("--itr_config", required=True)
    ap.add_argument("--itr_ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--tz_offset_hours", type=int, default=9)
    ap.add_argument("--outdir", default="results_compare_models")
    ap.add_argument("--save_arrays", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    arrays_dir = os.path.join(args.outdir, "arrays") if args.save_arrays else None

    mmk = run_one_model(
        name="MMK",
        config_path=args.mmk_config,
        ckpt_path=args.mmk_ckpt,
        split=args.split,
        batch_size=args.batch_size,
        tz_offset_hours=args.tz_offset_hours,
        save_arrays_dir=arrays_dir,
    )
    itr = run_one_model(
        name="iTr",
        config_path=args.itr_config,
        ckpt_path=args.itr_ckpt,
        split=args.split,
        batch_size=args.batch_size,
        tz_offset_hours=args.tz_offset_hours,
        save_arrays_dir=arrays_dir,
    )

    # ---- mean96h (raw/night0) ----
    true_mean_raw = mmk["true_raw"].mean(axis=0)
    mmk_pred_mean_raw = mmk["pred_raw"].mean(axis=0)
    itr_pred_mean_raw = itr["pred_raw"].mean(axis=0)

    true_mean_n0 = mmk["true_night0"].mean(axis=0)
    mmk_pred_mean_n0 = mmk["pred_night0"].mean(axis=0)
    itr_pred_mean_n0 = itr["pred_night0"].mean(axis=0)

    plot_mean96h_compare(
        true_mean_raw, mmk_pred_mean_raw, itr_pred_mean_raw,
        outpath=os.path.join(args.outdir, "compare_mean96h_pred_vs_true_raw.png"),
        title="96h Mean (RAW): Prediction mean vs True mean",
    )
    plot_mean96h_compare(
        true_mean_n0, mmk_pred_mean_n0, itr_pred_mean_n0,
        outpath=os.path.join(args.outdir, "compare_mean96h_pred_vs_true_night0.png"),
        title="96h Mean (Night0): Prediction mean vs True mean",
    )

    # ---- horizon mean (実質同じ計算だが、研究説明用に別名で保存) ----
    plot_horizon_mean_compare(
        true_mean_raw, mmk_pred_mean_raw, itr_pred_mean_raw,
        outpath=os.path.join(args.outdir, "compare_horizon_mean_raw.png"),
        title="Horizon Mean (RAW): avg over windows",
    )
    plot_horizon_mean_compare(
        true_mean_n0, mmk_pred_mean_n0, itr_pred_mean_n0,
        outpath=os.path.join(args.outdir, "compare_horizon_mean_night0.png"),
        title="Horizon Mean (Night0): avg over windows",
    )

    # ---- hour-of-day mean (t0が両方ある場合のみ) ----
    if mmk["t0"] is not None and itr["t0"] is not None:
        pred_len = int(mmk["pred_len"])
        # Trueは同一データであれば一致する前提でMMK側を代表
        true_hod_raw = hour_of_day_mean(mmk["t0"], mmk["true_raw"], pred_len, args.tz_offset_hours)
        mmk_hod_raw = hour_of_day_mean(mmk["t0"], mmk["pred_raw"], pred_len, args.tz_offset_hours)
        itr_hod_raw = hour_of_day_mean(itr["t0"], itr["pred_raw"], pred_len, args.tz_offset_hours)

        true_hod_n0 = hour_of_day_mean(mmk["t0"], mmk["true_night0"], pred_len, args.tz_offset_hours)
        mmk_hod_n0 = hour_of_day_mean(mmk["t0"], mmk["pred_night0"], pred_len, args.tz_offset_hours)
        itr_hod_n0 = hour_of_day_mean(itr["t0"], itr["pred_night0"], pred_len, args.tz_offset_hours)

        plot_hour_of_day_compare(
            true_hod_raw, mmk_hod_raw, itr_hod_raw,
            outpath=os.path.join(args.outdir, "compare_hour_of_day_mean_raw.png"),
            title="Hour-of-Day Mean (RAW, JST): Prediction mean vs True mean",
        )
        plot_hour_of_day_compare(
            true_hod_n0, mmk_hod_n0, itr_hod_n0,
            outpath=os.path.join(args.outdir, "compare_hour_of_day_mean_night0.png"),
            title="Hour-of-Day Mean (Night0, JST): Prediction mean vs True mean",
        )

    # ---- info ----
    info = {
        "split": args.split,
        "tz_offset_hours": args.tz_offset_hours,
        "batch_size": args.batch_size,
        "mmk": {k: mmk[k] for k in ["config", "ckpt", "data_dir", "pred_len", "has_t0", "has_marker_y"]},
        "itr": {k: itr[k] for k in ["config", "ckpt", "data_dir", "pred_len", "has_t0", "has_marker_y"]},
        "outputs": [
            "compare_mean96h_pred_vs_true_raw.png",
            "compare_mean96h_pred_vs_true_night0.png",
            "compare_horizon_mean_raw.png",
            "compare_horizon_mean_night0.png",
            "compare_hour_of_day_mean_raw.png (if t0 available)",
            "compare_hour_of_day_mean_night0.png (if t0 available)",
        ],
        "note": "This script does not modify plot_pred_vs_true_revin.py; it reproduces its logic to compare two models in one run."
    }
    with open(os.path.join(args.outdir, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("[OK] saved plots to:", args.outdir)


if __name__ == "__main__":
    main()
