# -*- coding: utf-8 -*-
"""
tools/plot_pred_vs_true_revin.py

目的:
- ckpt + config から split(train/val/test) の予測を出して outdir に .npy 保存
- marker_y による night0（夜間0化）を適用して保存
- t0 は data_dir/{split}_t0.npy があれば保存

対応:
- iTransformerPeak / iTransformer
- MMK_Mix
- それ以外も、easytsf/model/{model_name}.py を import し、
  クラス {model_name} or Model があればそれを使って生成する

出力（outdir）:
- pred_pv_raw.npy
- pred_pv_night0.npy
- pred_pv_test.npy      (互換名：night0 を保存)
- true_pv_test.npy
- test_marker_y.npy
- test_t0.npy
"""

import os
import argparse
import importlib
import importlib.util
import inspect
import numpy as np
import torch


# --------------------------
# config loader
# --------------------------
def load_exp_conf(config_path: str):
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)

    spec = importlib.util.spec_from_file_location("exp_conf_module", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "exp_conf"):
        raise AttributeError(f"exp_conf not found in config: {config_path}")
    return mod.exp_conf


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _normalize_marker_y_to_2d(marker_y, pred_len: int):
    """
    marker_y:
      - None
      - (N, pred_len)
      - (N, pred_len, 1)
    -> (N, pred_len) float32 or None
    """
    if marker_y is None:
        return None
    my = to_numpy(marker_y)
    if my.ndim == 3 and my.shape[-1] == 1:
        my = my[:, :, 0]
    elif my.ndim == 2:
        pass
    else:
        raise ValueError(f"marker_y shape not supported: {my.shape}")
    if my.shape[1] != pred_len:
        raise ValueError(f"marker_y pred_len mismatch: marker_y={my.shape}, pred_len={pred_len}")
    return my.astype(np.float32)


def apply_night0(pred_2d: np.ndarray, marker_y_2d: np.ndarray):
    """pred_2d: (N,pred_len), marker_y_2d: (N,pred_len) or None"""
    if marker_y_2d is None:
        return pred_2d
    return pred_2d * marker_y_2d


def strip_model_prefix(state_dict: dict):
    """Lightning runner の state_dict から base model 用に 'model.' prefix を外す"""
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        new_state[nk] = v
    return new_state


def load_ckpt_state_dict(ckpt_path: str):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        return sd["state_dict"]
    return sd


def _extract_pred_pv(pred, pv_index_if_multi: int):
    """
    pred:
      - (B,T)
      - (B,T,1)
      - (B,T,C)
    -> (B,T)
    """
    if pred.dim() == 2:
        return pred
    if pred.dim() == 3:
        C = pred.size(-1)
        if C == 1:
            return pred[:, :, 0]
        idx = pv_index_if_multi if C > pv_index_if_multi else 0
        return pred[:, :, idx]
    raise ValueError(f"Unexpected pred shape: {tuple(pred.shape)}")


def get_model(model_name: str, exp_conf: dict):
    """
    traingraph.py と同じ思想：
    - model_name からモデルクラスを特定
    - __init__ の引数に合わせて exp_conf をフィルタして生成
    """
    # よく使うモデル名は明示対応
    if model_name in ["iTransformerPeak", "iTransformer"]:
        from easytsf.model.iTransformer_peak import iTransformerPeak
        model_class = iTransformerPeak

    elif model_name == "MMK_Mix":
        from easytsf.model.MMK_Mix import MMK_Mix
        model_class = MMK_Mix

    else:
        # 汎用：easytsf.model.{model_name} を import し、{model_name} クラス or Model クラスを探す
        try:
            mod = importlib.import_module(f"easytsf.model.{model_name}")
        except Exception as e:
            raise ImportError(
                f"Failed to import module easytsf.model.{model_name}. "
                f"Check that easytsf/model/{model_name}.py exists.\nOriginal error: {e}"
            )

        if hasattr(mod, model_name):
            model_class = getattr(mod, model_name)
        elif hasattr(mod, "Model"):
            model_class = getattr(mod, "Model")
        else:
            # 最後の手段：モジュール内のクラスを探す（最初のクラスを採用）
            candidates = []
            for name, obj in vars(mod).items():
                if isinstance(obj, type):
                    candidates.append((name, obj))
            if not candidates:
                raise ImportError(f"No model class found in easytsf.model.{model_name}")
            model_class = candidates[0][1]

    sig = inspect.signature(model_class.__init__)
    valid_params = [p.name for p in sig.parameters.values() if p.name != "self"]
    filtered_conf = {k: v for k, v in exp_conf.items() if k in valid_params}

    return model_class(**filtered_conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--outdir", default="results_predplots")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    exp_conf = load_exp_conf(args.config)
    ensure_dir(args.outdir)

    # --------------------------
    # config values
    # --------------------------
    data_dir = exp_conf.get("data_dir", None)
    if data_dir is None:
        raise KeyError("exp_conf['data_dir'] not found in config.")

    model_name = exp_conf.get("model_name", None)
    if model_name is None:
        raise KeyError("exp_conf['model_name'] not found in config.")

    pred_len = int(exp_conf.get("pred_len", exp_conf.get("fut_len", 96)))
    ghi_index = int(exp_conf.get("ghi_index", 2))
    ghi_boost = float(exp_conf.get("ghi_boost", 1.0))
    pv_index = int(exp_conf.get("pv_index", 0))
    batch_size = int(exp_conf.get("batch_size", 64))

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    # --------------------------
    # DataModule（あなたの構成：easytsf.runner.data_runner）
    # --------------------------
    from easytsf.runner.data_runner import NPYDataInterface
    dm = NPYDataInterface(**exp_conf)
    dm.setup(stage=None)

    if args.split == "train":
        loader = dm.train_dataloader()
    elif args.split == "val":
        loader = dm.val_dataloader()
    else:
        loader = dm.test_dataloader()

    # --------------------------
    # Model build from model_name
    # --------------------------
    model = get_model(model_name, exp_conf)
    model.to(device)
    model.eval()

    # --------------------------
    # Load ckpt
    # --------------------------
    ckpt_path = os.path.abspath(args.ckpt)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    raw_sd = load_ckpt_state_dict(ckpt_path)
    base_sd = strip_model_prefix(raw_sd)
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    if missing:
        print("[WARN] missing keys:", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("[WARN] unexpected keys:", unexpected[:10], "..." if len(unexpected) > 10 else "")

    # --------------------------
    # Predict loop
    # --------------------------
    preds = []
    trues = []
    marker_ys = []

    with torch.no_grad():
        for batch in loader:
            # data_runner の Dataset が返す: (x, x_mark, y, marker_y)
            var_x, marker_x, var_y, marker_y = batch

            var_x = var_x.to(device).float()
            marker_x = marker_x.to(device).float()
            var_y = var_y.to(device).float()
            marker_y = marker_y.to(device).float()

            # GHI boost（traingraph.py と同じ思想）
            if var_x.dim() == 3 and var_x.size(-1) > ghi_index and ghi_boost != 1.0:
                var_x = var_x.clone()
                var_x[:, :, ghi_index] = var_x[:, :, ghi_index] * ghi_boost

            # forward（モデルによって引数が違うのでフォールバック）
            try:
                out = model(var_x, marker_x)
            except TypeError:
                out = model(var_x)

            if isinstance(out, (tuple, list)):
                out = out[0]

            # pred -> (B, pred_len)
            pred_2d = _extract_pred_pv(out, pv_index_if_multi=pv_index)
            pred_2d = pred_2d[:, -pred_len:]

            # true -> (B, pred_len)
            if var_y.dim() == 3 and var_y.size(-1) == 1:
                true_2d = var_y[:, -pred_len:, 0]
            elif var_y.dim() == 2:
                true_2d = var_y[:, -pred_len:]
            else:
                raise ValueError(f"Unexpected var_y shape: {tuple(var_y.shape)}")

            preds.append(to_numpy(pred_2d).astype(np.float32))
            trues.append(to_numpy(true_2d).astype(np.float32))
            marker_ys.append(to_numpy(marker_y))

    pred_raw = np.concatenate(preds, axis=0)   # (N, pred_len)
    true_2d = np.concatenate(trues, axis=0)    # (N, pred_len)

    marker_y_np = np.concatenate(marker_ys, axis=0) if len(marker_ys) > 0 else None
    marker_y_2d = _normalize_marker_y_to_2d(marker_y_np, pred_len=pred_len) if marker_y_np is not None else None
    pred_night0 = apply_night0(pred_raw, marker_y_2d)

    # t0（存在すれば）
    t0_path = os.path.join(data_dir, f"{args.split}_t0.npy")
    t0 = np.load(t0_path, allow_pickle=True) if os.path.exists(t0_path) else None

    # --------------------------
    # Save
    # --------------------------
    np.save(os.path.join(args.outdir, "pred_pv_raw.npy"), pred_raw.astype(np.float32))
    np.save(os.path.join(args.outdir, "pred_pv_night0.npy"), pred_night0.astype(np.float32))

    # 互換名
    np.save(os.path.join(args.outdir, "pred_pv_test.npy"), pred_night0.astype(np.float32))
    np.save(os.path.join(args.outdir, "true_pv_test.npy"), true_2d.astype(np.float32))

    if marker_y_np is not None:
        np.save(os.path.join(args.outdir, "test_marker_y.npy"), marker_y_np)
    if t0 is not None:
        np.save(os.path.join(args.outdir, "test_t0.npy"), t0)

    print("[OK] model_name:", model_name)
    print("[OK] saved to:", os.path.abspath(args.outdir))
    print("[INFO] pred_raw:", pred_raw.shape, "pred_night0:", pred_night0.shape, "true:", true_2d.shape)
    if marker_y_np is not None:
        print("[INFO] marker_y:", marker_y_np.shape, "marker_y_2d:", (None if marker_y_2d is None else marker_y_2d.shape))
    if t0 is not None:
        print("[INFO] t0:", t0.shape, t0.dtype)


if __name__ == "__main__":
    main()
