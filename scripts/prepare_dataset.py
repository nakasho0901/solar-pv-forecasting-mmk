# build_dataset_pv_windows_pvin.py
# ------------------------------------------------------------
# 24-24 / 48-24 などの窓切りデータセットを作る（PV入力あり版）
# 入力CSV: pv_weather_hourly.csv（time, pv_kwh, ghi_wm2, 時刻特徴, lag, rollなど）
#
# 出力（例: outdir/24-24_pvin/）
#   train_x.npy, train_y.npy, train_daymask_y.npy, train_t0.npy
#   val_x.npy,   val_y.npy,   val_daymask_y.npy,   val_t0.npy
#   test_x.npy,  test_y.npy,  test_daymask_y.npy,  test_t0.npy
#   meta.json（列情報/窓設定）
#   scaler.json（trainでfitした標準化パラメータ）
#
# 重要：
# - PV(pv_kwh) を入力Xにも含める（MMK_Mixと整合）
# - feature順の先頭を pv_kwh に固定（pv_index=0 を確定）
# - 窓は「1時間連続」区間のみ採用（時間ギャップ跨ぎは捨てる）
# ------------------------------------------------------------

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
    name: str  # "24-24_pvin" など


def parse_make_list(make_list: List[str], suffix: str) -> List[WindowSpec]:
    specs = []
    for s in make_list:
        a, b = s.split("-")
        in_len = int(a)
        out_len = int(b)
        specs.append(WindowSpec(in_len_h=in_len, out_len_h=out_len, name=f"{in_len}-{out_len}{suffix}"))
    return specs


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def compute_time_gap_mask(time_like: Union[pd.Series, pd.DatetimeIndex, np.ndarray, List]) -> np.ndarray:
    """
    各行が「前行から1時間差で連続しているか」を判定。
    - 先頭行は True 扱い
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
    - y: (N, out_len)  ※ yは target_col（pv_kwh）
    - daymask_y: (N, out_len)
    - t0: (N,)  ※ 出力開始時刻
    """
    time_series = pd.to_datetime(df["time"])
    cont_ok = compute_time_gap_mask(time_series)

    values_X = df[feature_cols].to_numpy(dtype=np.float32)
    values_y = df[target_col].to_numpy(dtype=np.float32)
    values_day = df[daylight_col].to_numpy(dtype=np.float32)

    X_list, y_list, m_list, t_list = [], [], [], []
    n = len(df)

    for i in range(in_len - 1, n - out_len, stride):
        in_start = i - (in_len - 1)
        out_start = i + 1
        out_end = i + out_len

        # 連続性：in_start..out_end が 1時間刻み
        if not cont_ok[in_start + 1 : out_end + 1].all():
            continue

        X = values_X[in_start : i + 1]            # (in_len, F)
        y = values_y[out_start : out_end + 1]     # (out_len,)
        m = values_day[out_start : out_end + 1]   # (out_len,)
        t0 = time_series.iloc[out_start]

        X_list.append(X)
        y_list.append(y)
        m_list.append(m)
        t_list.append(np.datetime64(pd.Timestamp(t0).to_datetime64()))

    X_arr = np.stack(X_list, axis=0) if X_list else np.zeros((0, in_len, len(feature_cols)), dtype=np.float32)
    y_arr = np.stack(y_list, axis=0) if y_list else np.zeros((0, out_len), dtype=np.float32)
    m_arr = np.stack(m_list, axis=0) if m_list else np.zeros((0, out_len), dtype=np.float32)
    t_arr = np.array(t_list, dtype="datetime64[ns]") if t_list else np.array([], dtype="datetime64[ns]")

    return X_arr, y_arr, m_arr, t_arr


def time_split_indices(n_samples: int, train_ratio: float, val_ratio: float) -> Dict[str, np.ndarray]:
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    if n_test < 1:
        raise ValueError(f"split ratio results in no test samples: n={n_samples}")
    idx = np.arange(n_samples)
    return {
        "train": idx[:n_train],
        "val": idx[n_train : n_train + n_val],
        "test": idx[n_train + n_val :],
    }


def fit_standard_scaler(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] == 0:
        raise ValueError("No samples to fit scaler.")
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standard_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean[None, None, :]) / std[None, None, :]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--make", type=str, nargs="+", default=["24-24"])
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)

    ap.add_argument("--target_col", type=str, default="pv_kwh")     # yはこれ
    ap.add_argument("--daylight_col", type=str, default="is_daylight")
    ap.add_argument("--daylight_from_ghi", action="store_true")

    ap.add_argument("--pv_input_col", type=str, default="pv_kwh")   # Xにも入れるPV列（基本はpv_kwh）
    ap.add_argument("--drop_cols", type=str, nargs="*", default=[])

    # 出力ディレクトリ名を区別するためのsuffix
    ap.add_argument("--suffix", type=str, default="_pvin")

    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    if "time" not in df.columns:
        raise ValueError("CSV must contain 'time' column.")
    if args.target_col not in df.columns:
        raise ValueError(f"target_col '{args.target_col}' not found.")
    if args.pv_input_col not in df.columns:
        raise ValueError(f"pv_input_col '{args.pv_input_col}' not found.")

    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    # daymask
    if args.daylight_col not in df.columns:
        if "ghi_wm2" not in df.columns:
            raise ValueError("No is_daylight column and no ghi_wm2 to compute it.")
        df[args.daylight_col] = (df["ghi_wm2"] > 0).astype(np.float32)
    if args.daylight_from_ghi:
        if "ghi_wm2" not in df.columns:
            raise ValueError("daylight_from_ghi requires ghi_wm2 column.")
        df[args.daylight_col] = (df["ghi_wm2"] > 0).astype(np.float32)

    # ---- feature列を作る（PVを先頭に固定）----
    # まず「それ以外の列」を作る
    # ※ target_col(pv_kwh) も入力に含めるので exclude から外す（ここが前回と違う）
    exclude = {"time"}  # target_colは除外しない
    for c in args.drop_cols:
        exclude.add(c)

    # 候補（pv_input_colも含まれるが、後で先頭に移動する）
    candidate = [c for c in df.columns if c not in exclude]

    # pv_input_col を先頭に固定し、残りは元の順序で重複なく並べる
    feature_cols = [args.pv_input_col] + [c for c in candidate if c != args.pv_input_col]

    print("[INFO] rows:", len(df), "cols:", len(df.columns))
    print("[INFO] target_col:", args.target_col)
    print("[INFO] daylight_col:", args.daylight_col)
    print("[INFO] pv_input_col (X first):", args.pv_input_col)
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
                f"Try smaller stride (e.g., 1) or check time gaps."
            )

        split = time_split_indices(X.shape[0], args.train_ratio, args.val_ratio)

        mean, std = fit_standard_scaler(X[split["train"]])
        Xs = apply_standard_scaler(X, mean, std)

        def save_split(name: str, idx: np.ndarray):
            np.save(os.path.join(out_base, f"{name}_x.npy"), Xs[idx])
            np.save(os.path.join(out_base, f"{name}_y.npy"), y[idx])
            np.save(os.path.join(out_base, f"{name}_daymask_y.npy"), m[idx])
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
                "PV(pv_kwh) is included in X as the first feature (pv_index=0).",
                "Windows that cross time gaps (not 1-hour continuous) are excluded.",
                "Scaling is fit on train split only and applied to all splits.",
                "daymask_y can be used for day-only loss/evaluation (night weight=0).",
            ],
        }
        with open(os.path.join(out_base, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        scaler = {"mean": mean.tolist(), "std": std.tolist(), "feature_cols": feature_cols}
        with open(os.path.join(out_base, "scaler.json"), "w", encoding="utf-8") as f:
            json.dump(scaler, f, ensure_ascii=False, indent=2)

        # 便利情報：pv_index と ghi_index をログで出す
        pv_index = feature_cols.index(args.pv_input_col)
        ghi_index = feature_cols.index("ghi_wm2") if "ghi_wm2" in feature_cols else None
        print(f"[INFO] pv_index={pv_index} (feature '{args.pv_input_col}')")
        print(f"[INFO] ghi_index={ghi_index} (feature 'ghi_wm2')")

        print(f"[OK] saved to: {out_base}")


if __name__ == "__main__":
    main()
