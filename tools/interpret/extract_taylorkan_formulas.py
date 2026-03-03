# -*- coding: utf-8 -*-
r"""
extract_taylorkan_formulas.py

目的:
- MMK_FusionPV_FeatureToken_v2 の ckpt をロード
- 各層の experts を走査し、TaylorKANLayer の coeffs/bias を抽出
- 発表で扱えるように「重要度ランキング」「上位項だけの式」を JSON/NPZ に保存

使い方(Windows; 1文ワンライナー):
set PYTHONPATH=C:\Users\nakas\my_project\solar-kan && python extract_taylorkan_formulas.py --ckpt "C:\Users\nakas\my_project\solar-kan\save\MMK_FusionPV_FeatureToken\version_12\checkpoints\last.ckpt" --model_py "C:\Users\nakas\my_project\solar-kan\easytsf\model\MMK_FusionPV_FeatureToken_v2.py" --config_py "C:\Users\nakas\my_project\solar-kan\config\tsukuba_conf\MMK_FusionPV_FeatureToken_A11_stride1_v2.py" --meta_json "C:\Users\nakas\my_project\solar-kan\dataset_PV\prepared\96-96_fusionA11_96_96_stride1\meta.json" --outdir "C:\Users\nakas\my_project\solar-kan\results_taylorkan_export"
"""

import os
import json
import argparse
import importlib.util
from typing import Any, Dict, List

import numpy as np
import torch

# ここは「プロジェクト内 import」を使う（PYTHONPATHが通っている前提）
from easytsf.layer.kanlayer import TaylorKANLayer


# -------------------------
# 1) ファイルパスから module を読む
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
# 2) Lightning ckpt / plain state_dict 両対応
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


# -------------------------
# 3) TaylorKAN 係数の要約
# -------------------------
def summarize_taylor_coeffs(coeffs: torch.Tensor, bias: torch.Tensor | None, topk_terms: int = 20) -> Dict[str, Any]:
    """
    coeffs: (out_dim, in_dim, order)
    bias:   (1, out_dim) or (out_dim,) or None
    """
    c = coeffs.detach().cpu().float().numpy()
    O, I, order = c.shape

    bias_vec = None
    if bias is not None:
        b = bias.detach().cpu().float()
        if b.ndim == 2 and b.shape[0] == 1:
            b = b[0]
        bias_vec = b.numpy()

    # 入力次元ごとの寄与（ノルム）
    in_norm = np.sqrt((c ** 2).sum(axis=(0, 2)))  # (I,)
    out_norm = np.sqrt((c ** 2).sum(axis=(1, 2))) # (O,)
    in_rank = np.argsort(-in_norm)
    out_rank = np.argsort(-out_norm)

    # 係数の絶対値が大きい項 topk
    flat = np.abs(c).reshape(-1)
    topk = min(topk_terms, flat.size)
    idx = np.argpartition(-flat, topk - 1)[:topk]
    idx = idx[np.argsort(-flat[idx])]

    top_terms = []
    for flat_i in idx:
        o = flat_i // (I * order)
        rem = flat_i % (I * order)
        i = rem // order
        k = rem % order
        coef_val = float(c[o, i, k])
        top_terms.append({
            "out_dim": int(o),
            "in_dim": int(i),
            "power": int(k),
            "coef": coef_val,
            "abs_coef": float(abs(coef_val)),
        })

    # 対角成分だけの近似（o==iのみ）
    diag_terms = []
    for d in range(min(O, I)):
        for k in range(order):
            v = float(c[d, d, k])
            if abs(v) > 0:
                diag_terms.append({"dim": int(d), "power": int(k), "coef": v})

    return {
        "shape": {"out_dim": int(O), "in_dim": int(I), "order": int(order)},
        "bias": None if bias_vec is None else bias_vec.tolist(),
        "in_norm": in_norm.tolist(),
        "out_norm": out_norm.tolist(),
        "in_rank": in_rank.tolist(),
        "out_rank": out_rank.tolist(),
        "top_terms": top_terms,
        "diag_terms": diag_terms,
    }


def format_sparse_polynomial(top_terms: List[Dict[str, Any]], out_dim: int, max_terms: int = 8) -> str:
    terms = [t for t in top_terms if t["out_dim"] == out_dim][:max_terms]
    if not terms:
        return f"y[{out_dim}] ≈ (no large terms found)"
    parts = []
    for t in terms:
        coef = t["coef"]
        i = t["in_dim"]
        k = t["power"]
        if k == 0:
            parts.append(f"{coef:+.3e}")
        elif k == 1:
            parts.append(f"{coef:+.3e} * x[{i}]")
        else:
            parts.append(f"{coef:+.3e} * (x[{i}]**{k})")
    return f"y[{out_dim}] ≈ " + " ".join(parts)


# -------------------------
# 4) main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_py", required=True)
    ap.add_argument("--config_py", required=True)
    ap.add_argument("--meta_json", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--topk_terms", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # config 読み込み
    conf_mod = load_module_from_path("exp_conf_mod", args.config_py)
    exp_conf = conf_mod.exp_conf

    # model class 読み込み
    model_mod = load_module_from_path("model_mod", args.model_py)
    ModelClass = getattr(model_mod, exp_conf["model_class"])

    # meta 読み込み
    with open(args.meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # モデル生成（configに合わせる）
    model = ModelClass(
        hist_len=int(exp_conf["hist_len"]),
        pred_len=int(exp_conf["pred_len"]),
        var_num=int(exp_conf["var_num"]),
        pv_index=int(exp_conf.get("pv_index", 0)),
        token_dim=int(exp_conf["token_dim"]),
        layer_num=int(exp_conf["layer_num"]),
        layer_hp=exp_conf["layer_hp"],
        use_layernorm=bool(exp_conf["use_layernorm"]),
        dropout=float(exp_conf["dropout"]),
        fusion=str(exp_conf["fusion"]),
        baseline_mode=str(exp_conf["baseline_mode"]),
        use_pv_revin=bool(exp_conf["use_pv_revin"]),
        revin_eps=float(exp_conf["revin_eps"]),
        enforce_nonneg=bool(exp_conf["enforce_nonneg"]),
    )

    # ckpt ロード
    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = strip_prefix_if_needed(extract_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(sd, strict=False)

    model.eval()

    results = {
        "ckpt": args.ckpt,
        "model_py": args.model_py,
        "config_py": args.config_py,
        "meta_json": args.meta_json,
        "load_state_dict": {"missing": missing, "unexpected": unexpected},
        "meta_feature_cols": meta.get("columns", {}).get("feature_cols", []),
        "layers": [],
    }

    # 各層を走査して TaylorKANLayer を持つ expert を抽出
    for li, blk in enumerate(model.blocks):
        layer_info = {"layer": li, "taylorkan_experts": []}
        mok = blk.mok

        for ei, ex in enumerate(mok.experts):
            tr = getattr(ex, "transform", None)
            if tr is None:
                continue

            # ここが本質：TaylorKANLayer かどうか
            if not isinstance(tr, TaylorKANLayer):
                continue

            coeffs = tr.coeffs
            bias = tr.bias if hasattr(tr, "bias") else None

            summ = summarize_taylor_coeffs(coeffs, bias, topk_terms=args.topk_terms)

            # 発表用：寄与が大きい出力次元3つの簡易式
            out_rank = summ["out_rank"]
            sample_outs = out_rank[:3] if len(out_rank) >= 3 else out_rank
            pretty = [format_sparse_polynomial(summ["top_terms"], int(od), max_terms=8) for od in sample_outs]

            layer_info["taylorkan_experts"].append({
                "expert_id": ei,
                "summary": summ,
                "pretty_equations_top3_outdims": pretty,
            })

            # 係数自体も保存（巨大なので expert ごと）
            npz_path = os.path.join(args.outdir, f"layer{li}_expert{ei}_taylorkan_coeffs.npz")
            np.savez_compressed(
                npz_path,
                coeffs=coeffs.detach().cpu().numpy(),
                bias=None if bias is None else bias.detach().cpu().numpy(),
            )

        results["layers"].append(layer_info)

    json_path = os.path.join(args.outdir, "taylorkan_export_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("[OK] saved:", args.outdir)
    print(" -", json_path)
    print(" - coeffs npz: layer*_expert*_taylorkan_coeffs.npz")
    print("[load_state_dict] missing:", len(missing), "unexpected:", len(unexpected))

    # 成功チェック用：TaylorKAN expert が1つでもあったか
    found = sum(len(L["taylorkan_experts"]) for L in results["layers"])
    print("[check] TaylorKAN experts found:", found)


if __name__ == "__main__":
    main()
