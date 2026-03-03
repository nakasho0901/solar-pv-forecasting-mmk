# -*- coding: utf-8 -*-
"""
build_dataset_pv_windows_noscale_itr.py

目的：
- iTransformer用に「前処理スケーリングなし（noscale）」の窓切りnpyを作る
- 縦軸を pv_kwh（実スケール）で比較できるようにする
- 特徴量は "シンプル8個" をデフォルト採用（公平比較・解釈性優先）

入力CSV想定（例）:
dataset_PV/prepared/pv_weather_hourly.csv
  time, pv_kwh, temp_c, rh_pct, ghi_wm2, hour_sin, hour_cos, dow_sin, dow_cos, (optional is_daylight, lag_*, roll_* ...)

出力:
outdir/<name>/
  train_x.npy, train_y.npy, train_marker_y.npy, train_t0.npy
  val_x.npy,   val_y.npy,   val_marker_y.npy,   val_t0.npy
  test_x.npy,  test_y.npy,  test_marker_y.npy,  test_t0.npy
  meta.json
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Union

import numpy as np
import pandas as pd


@dataclass
class WindowSpec:
    in_len_h: int
    out_len_h: int
    name: str


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_make_list(make_list: List[str], suffix: str) -> List[WindowSpec]:
    specs = []
    for s in make_list:
        a, b = s.split("-")
        in_len = int(a)
        out_len = int(b)
        specs.append(WindowSpec(in_len_h=in_len, out_len_h=out_len, name=f"{in_len}-{out_len}{suffix}"))
    return specs


def compute_time_gap_mask(time_like: Union[pd.Series, pd.DatetimeIndex, np.ndarray, List]) -> np.ndarray:
    """
    各行が「前行から1時間差で連続しているか」を判定。
    - 先頭行は True
    """
    t = pd.to_datetime(time_like)
    if isinstance(t, pd.DatetimeIndex):
        ts = pd.Series(t.values)
    elif isinstance(t, pd.Series):
        ts = t
    else:
        ts = pd.Series(t)

    dt = ts.diff().dt.total_seconds().to_numpy()
    ok = np.ones(len(ts), dtype=bool)
    ok[1:] = np.isclose(dt[1:], 3600.0)
    return ok


def build_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    daylight_col: str,
    in_len: int,
    out_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    - X: (N, in_len, F)
    - y: (N, out_len)          ※ターゲットはpv_kwhのみ（1系列）
    - m: (N, out_len)          ※daylightマスク（0/1）
    - t0: (N,) datetime64[ns]  ※窓の開始時刻
    """
    times = pd.to_datetime(df["time"]).reset_index(drop=True)
    cont_mask = compute_time_gap_mask(times)

    X_list, y_list, m_list, t0_list = [], [], [], []

    values = df[feature_cols].to_numpy(dtype=np.float32)
    yvals = df[target_col].to_numpy(dtype=np.float32)
    daymask = df[daylight_col].to_numpy(dtype=np.float32)

    total = len(df)
    end = total - (in_len + out_len) + 1
    for s in range(0, end, stride):
        in_slice = slice(s, s + in_len)
        out_slice = slice(s + in_len, s + in_len + out_len)

        # 1時間連続の区間だけ採用（ギャップ跨ぎは捨てる）
        if not cont_mask[s:s + in_len + out_len].all():
            continue

        X_list.append(values[in_slice])
        y_list.append(yvals[out_slice])
        m_list.append(daymask[out_slice])
        t0_list.append(times.iloc[s].to_datetime64())

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    m = np.stack(m_list, axis=0)
    t0 = np.array(t0_list)
    return X, y, m, t0


def time_split_indices(n: int, train_ratio: float, val_ratio: float) -> Dict[str, np.ndarray]:
    """
    時系列順で train / val / test を分割
    """
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    idx = np.arange(n)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--make", type=str, nargs="+", default=["96-96"])
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)

    ap.add_argument("--target_col", type=str, default="pv_kwh")
    ap.add_argument("--daylight_col", type=str, default="is_daylight")
    ap.add_argument("--daylight_from_ghi", action="store_true")

    # 8特徴量（公平比較・王道）
    ap.add_argument("--simple8", action="store_true", help="Use 8 simple features only (recommended).")
    ap.add_argument("--suffix", type=str, default="_itr_noscale")

    # 任意で列を落とす（例: is_daylight をXに入れたくない等）
    ap.add_argument("--drop_cols", type=str, nargs="*", default=[])

    # hardzero（夜はPVを0固定）を入れたい場合
    ap.add_argument("--hardzero", action="store_true", help="Force pv_kwh=0 when is_daylight==0")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "time" not in df.columns:
        raise ValueError("CSV must contain 'time' column.")
    if args.target_col not in df.columns:
        raise ValueError(f"target_col '{args.target_col}' not found in csv.")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # daylight列が無いならghiから作る
    if args.daylight_col not in df.columns:
        if "ghi_wm2" not in df.columns:
            raise ValueError("No is_daylight column and no ghi_wm2 to compute it.")
        df[args.daylight_col] = (df["ghi_wm2"] > 0).astype(np.float32)
    if args.daylight_from_ghi:
        if "ghi_wm2" not in df.columns:
            raise ValueError("daylight_from_ghi requires ghi_wm2 column.")
        df[args.daylight_col] = (df["ghi_wm2"] > 0).astype(np.float32)

    # hardzero（必要なら）
    if args.hardzero:
        df.loc[df[args.daylight_col] <= 0, args.target_col] = 0.0

    # iTransformer用の「シンプル8特徴量」
    # pv_kwh をXの先頭に置く（pv_index=0 で固定できて事故らない）
    simple8_cols = ["pv_kwh", "temp_c", "rh_pct", "ghi_wm2", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]

    if args.simple8:
        for c in simple8_cols:
            if c not in df.columns:
                raise ValueError(f"[ERROR] simple8 requires column '{c}' but not found in csv.")
        feature_cols = simple8_cols.copy()
    else:
        # simple8を使わない場合：time以外を全部入れる（ただしdrop_colsは除外）
        exclude = {"time"}
        for c in args.drop_cols:
            exclude.add(c)
        candidate = [c for c in df.columns if c not in exclude]
        # PVを先頭固定
        feature_cols = ["pv_kwh"] + [c for c in candidate if c != "pv_kwh"]

    # drop_cols で明示的に落としたい列が feature_cols に混ざってたら除外
    feature_cols = [c for c in feature_cols if c not in set(args.drop_cols)]

    print("[INFO] rows:", len(df), "cols:", len(df.columns))
    print("[INFO] target_col:", args.target_col)
    print("[INFO] daylight_col:", args.daylight_col)
    print("[INFO] num_features:", len(feature_cols))
    print("[INFO] features:", feature_cols)

    ensure_dir(args.outdir)

    specs = parse_make_list(args.make, suffix=args.suffix)
    for spec in specs:
        out_base = os.path.join(args.outdir, spec.name)
        ensure_dir(out_base)

        X, y, m, t0 = build_windows(
            df=df,
            feature_cols=feature_cols,
            target_col=args.target_col,
            daylight_col=args.daylight_col,
            in_len=spec.in_len_h,
            out_len=spec.out_len_h,
            stride=args.stride,
        )

        print(f"[INFO] {spec.name} windows:", X.shape, y.shape, m.shape, "start_time:", t0.shape)

        if X.shape[0] < 10:
            raise ValueError(
                f"Too few samples for {spec.name}: {X.shape[0]}. "
                f"Try smaller stride (e.g., 6 or 1) or check time gaps."
            )

        split = time_split_indices(X.shape[0], args.train_ratio, args.val_ratio)

        def save_split(name: str, idx: np.ndarray):
            np.save(os.path.join(out_base, f"{name}_x.npy"), X[idx])
            np.save(os.path.join(out_base, f"{name}_y.npy"), y[idx])
            np.save(os.path.join(out_base, f"{name}_marker_y.npy"), m[idx])
            np.save(os.path.join(out_base, f"{name}_t0.npy"), t0[idx])

        save_split("train", split["train"])
        save_split("val", split["val"])
        save_split("test", split["test"])

        meta = {
            "spec": {"in_len_h": spec.in_len_h, "out_len_h": spec.out_len_h, "stride_h": args.stride},
            "columns": {
                "feature_cols": feature_cols,
                "target_col": args.target_col,
                "daylight_col": args.daylight_col,
            },
            "split": {
                "train_ratio": args.train_ratio,
                "val_ratio": args.val_ratio,
                "counts": {k: int(v.size) for k, v in split.items()},
            },
            "notes": [
                "No scaling is applied (noscale). iTransformerPeak normalizes per-sample inside the model.",
                "PV(pv_kwh) is included in X as the first feature (pv_index=0).",
                "Windows that cross time gaps (not 1-hour continuous) are excluded.",
                "marker_y corresponds to daylight_col over the prediction horizon.",
                f"simple8={bool(args.simple8)} hardzero={bool(args.hardzero)}",
            ],
        }
        with open(os.path.join(out_base, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        pv_index = feature_cols.index("pv_kwh")
        ghi_index = feature_cols.index("ghi_wm2") if "ghi_wm2" in feature_cols else None
        print(f"[INFO] pv_index={pv_index} (feature 'pv_kwh')")
        print(f"[INFO] ghi_index={ghi_index} (feature 'ghi_wm2')")
        print(f"[OK] saved to: {out_base}")


if __name__ == "__main__":
    main()
