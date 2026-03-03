# -*- coding: utf-8 -*-
r"""
extract_basis_response_same_location.py

目的：
- 学習済み ckpt をロードしてモデルを復元
- 指定した layer_index / expert_id の expert.transform を取り出す
- 「同じ場所（同じtransform、同じtoken次元）」で
  1次元入力 x を掃引し、transform の出力応答 y を保存する

重要：
- SplineKANLayer / WaveletKANLayer / JacobiKANLayer などのクラス名に依存しない
  （環境によってクラス名が違っても動く）
- transform がどんなKAN基底でも「関数として評価」して (x, y) を出力できる

出力：
- outdir/<TransformClassName>_L{layer}_E{expert}_i{token}.npz
  中身：x, y, meta

実行例（Windows 1行）：
set PYTHONPATH=C:\Users\nakas\my_project\solar-kan && python extract_basis_response_same_location.py --ckpt "...\last.ckpt" --model_py "...\MMK_FusionPV_FeatureToken_v2.py" --config_py "...\MMK_FusionPV_FeatureToken_A11_stride1_v2.py" --layer_index 2 --expert_id 1 --token_index 12 --outdir results_basis_compare --xmin -2 --xmax 2 --n_points 400 --device cpu
"""

import os
import argparse
import importlib.util
from typing import Any, Dict

import numpy as np
import torch


# -------------------------
# util: module loader
# -------------------------
def load_module_from_path(mod_name: str, path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"module file not found: {path}")
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -------------------------
# util: ckpt -> state_dict (Lightning対応)
# -------------------------
def extract_state_dict(ckpt_obj: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        return ckpt_obj["state_dict"]
    if isinstance(ckpt_obj, dict) and all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
        return ckpt_obj
    raise ValueError("Unsupported checkpoint format: cannot find state_dict.")


def strip_prefix_if_needed(state_dict: Dict[str, torch.Tensor], prefixes=("model.", "net.", "module.")) -> Dict[str, torch.Tensor]:
    keys = list(state_dict.keys())
    for p in prefixes:
        if any(k.startswith(p) for k in keys):
            return {k[len(p):] if k.startswith(p) else k: v for k, v in state_dict.items()}
    return state_dict


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_py", required=True)
    ap.add_argument("--config_py", required=True)

    ap.add_argument("--layer_index", type=int, default=2)
    ap.add_argument("--expert_id", type=int, default=0)
    ap.add_argument("--token_index", type=int, default=0)

    ap.add_argument("--xmin", type=float, default=-2.0)
    ap.add_argument("--xmax", type=float, default=2.0)
    ap.add_argument("--n_points", type=int, default=400)

    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- device
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # ---- load config / model
    conf_mod = load_module_from_path("exp_conf_mod", args.config_py)
    exp_conf = conf_mod.exp_conf

    model_mod = load_module_from_path("model_mod", args.model_py)
    ModelClass = getattr(model_mod, exp_conf["model_class"])

    # NOTE: exp_conf のキーが足りない環境がありうるので、get で安全側に倒す
    model = ModelClass(
        hist_len=int(exp_conf["hist_len"]),
        pred_len=int(exp_conf["pred_len"]),
        var_num=int(exp_conf["var_num"]),
        pv_index=int(exp_conf.get("pv_index", 0)),
        token_dim=int(exp_conf["token_dim"]),
        layer_num=int(exp_conf["layer_num"]),
        layer_hp=exp_conf["layer_hp"],
        use_layernorm=bool(exp_conf.get("use_layernorm", True)),
        dropout=float(exp_conf.get("dropout", 0.0)),
        fusion=str(exp_conf.get("fusion", "FusionPV")),
        baseline_mode=str(exp_conf.get("baseline_mode", "FeatureToken")),
        use_pv_revin=bool(exp_conf.get("use_pv_revin", False)),
        revin_eps=float(exp_conf.get("revin_eps", 1e-5)),
        enforce_nonneg=bool(exp_conf.get("enforce_nonneg", False)),
    )

    # ---- load ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = strip_prefix_if_needed(extract_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(sd, strict=False)

    model.to(device)
    model.eval()

    # ---- locate transform
    try:
        blk = model.blocks[args.layer_index]
        mok = blk.mok
        ex = mok.experts[args.expert_id]
        tr = getattr(ex, "transform", None)
    except Exception as e:
        raise RuntimeError(f"Failed to locate layer/block/expert: {e}")

    if tr is None:
        raise RuntimeError("expert.transform is None (unexpected).")

    # ---- build input sweep (B, token_dim)
    token_dim = int(exp_conf["token_dim"])
    if not (0 <= args.token_index < token_dim):
        raise ValueError(f"token_index out of range: {args.token_index} (token_dim={token_dim})")

    xs = torch.linspace(args.xmin, args.xmax, int(args.n_points), device=device, dtype=torch.float32)
    X = torch.zeros((xs.numel(), token_dim), device=device, dtype=torch.float32)
    X[:, args.token_index] = xs

    # ---- forward through transform only
    Y = tr(X)

    # 期待形状： (B, token_dim) が多い。違う場合でも最低限 2D に落とす
    if Y.ndim == 3:
        # (B, T, D) 想定 -> 平均して (B, D)
        Y = Y.mean(dim=1)
    if Y.ndim != 2 or Y.shape[0] != X.shape[0]:
        raise RuntimeError(f"Unexpected transform output shape: {tuple(Y.shape)} (input {tuple(X.shape)})")

    y = Y[:, args.token_index].detach().cpu().numpy()
    x = xs.detach().cpu().numpy()

    # ---- save
    tname = type(tr).__name__
    out_path = os.path.join(
        args.outdir,
        f"{tname}_L{args.layer_index}_E{args.expert_id}_i{args.token_index}.npz"
    )
    np.savez_compressed(
        out_path,
        x=x,
        y=y,
        meta=np.array([{
            "layer_index": args.layer_index,
            "expert_id": args.expert_id,
            "token_index": args.token_index,
            "xmin": args.xmin,
            "xmax": args.xmax,
            "n_points": int(args.n_points),
            "transform_class": tname,
        }], dtype=object),
        load_state_missing=np.array(missing, dtype=object),
        load_state_unexpected=np.array(unexpected, dtype=object),
    )

    print("[OK] saved:", out_path)
    print("[info] transform:", tname)
    print("[load_state_dict] missing:", len(missing), "unexpected:", len(unexpected))


if __name__ == "__main__":
    main()
