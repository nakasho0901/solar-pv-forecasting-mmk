# tools/make_gate_heatmap_featuretoken.py
# -*- coding: utf-8 -*-
r"""
MMK_FusionPV_FeatureToken 用
- 入力: var_x (B, 96, 9)
- gate: 各層ごとに (B, F, E) を返す（return_gate=True）
- ヒートマップ: mean over batch -> (F, E) -> 転置して (E, F) を保存

使い方（例）:
set PYTHONPATH=C:\Users\nakas\my_project\solar-kan && python tools\make_gate_heatmap_featuretoken.py ^
  -c config\tsukuba_conf\MMK_FusionPV_Flatten_A9_96.py ^
  --ckpt "C:\Users\nakas\my_project\solar-kan\save\MMK_FusionPV_FeatureToken\version_0\checkpoints\last.ckpt" ^
  --split val --dataset_dir "C:\Users\nakas\my_project\solar-kan\dataset_PV\prepared\96-96_fusionA9_96_96_stride4" ^
  --outdir results_gate_featuretoken --batch_size 64 --window_index 0 --device auto
"""

import argparse
import importlib.util
import json
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt


# -------------------------
# config 読み込み（.pyファイル想定）
# -------------------------
def load_config_py(py_path: str):
    abs_path = os.path.abspath(py_path)
    mod_name = os.path.splitext(os.path.basename(abs_path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, abs_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import config: {abs_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


# -------------------------
# meta.json から feature 名を読む
# -------------------------
def load_feature_names(meta_json_path: str) -> List[str]:
    with open(meta_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cols = meta["columns"]["feature_cols"]
    if not isinstance(cols, list) or not all(isinstance(x, str) for x in cols):
        raise ValueError("meta.json の columns.feature_cols が想定外です。")
    return cols


# -------------------------
# ckpt ロード（Lightning形式 / state_dict 形式 両対応）
# -------------------------
def load_ckpt_to_model(model: torch.nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Lightning: {"state_dict": ...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
        # "model." prefix を外す
        new_state = {}
        for k, v in state.items():
            nk = k[6:] if k.startswith("model.") else k
            new_state[nk] = v
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(f"[INFO] ckpt loaded (lightning). missing={len(missing)} unexpected={len(unexpected)}")
        return

    # 素の state_dict
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        print(f"[INFO] ckpt loaded (state_dict). missing={len(missing)} unexpected={len(unexpected)}")
        return

    raise RuntimeError("ckpt 形式が想定外でした（torch.load の中身を確認してください）。")


# -------------------------
# ヒートマップ保存
# -------------------------
def save_heatmap(ev: np.ndarray, out_png: str, title: str, feature_names: List[str]) -> None:
    """
    ev: (E, F)
    """
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig = plt.figure(figsize=(max(8.0, ev.shape[1] * 0.7), max(4.5, ev.shape[0] * 0.6)))
    ax = fig.add_subplot(111)
    im = ax.imshow(ev, aspect="auto")

    ax.set_title(title)
    ax.set_ylabel("Expert ID")
    ax.set_xlabel("Feature")

    # feature 名を表示（多いと潰れるので間引き）
    F = len(feature_names)
    if F == ev.shape[1]:
        step = 1 if F <= 18 else int(np.ceil(F / 18))
        ticks = np.arange(0, F, step)
        ax.set_xticks(ticks)
        ax.set_xticklabels([feature_names[i] for i in ticks], rotation=45, ha="right")
    else:
        ax.set_xticks([])

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -------------------------
# main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="config .py path")
    parser.add_argument("--ckpt", required=True, help="checkpoint path")
    parser.add_argument("--dataset_dir", required=True, help="dataset directory (contains *_x.npy, meta.json)")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="which split x to use")
    parser.add_argument("--outdir", default="results_gate_featuretoken", help="output directory")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--window_index", type=int, default=0, help="which window to use (index into *_x.npy)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    # device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[INFO] device={device}")

    # config load
    cfg = load_config_py(args.config)
    if not hasattr(cfg, "exp_conf"):
        raise RuntimeError("config に exp_conf が見つかりません。")
    exp_conf: Dict = cfg.exp_conf

    # feature names
    meta_json_path = os.path.join(args.dataset_dir, "meta.json")
    feature_names = load_feature_names(meta_json_path)

    # model import & build
    # config の model_name は "MMK_FusionPV_FeatureToken" :contentReference[oaicite:4]{index=4}
    from easytsf.model.MMK_FusionPV_FeatureToken import MMK_FusionPV_FeatureToken  # type: ignore

    model = MMK_FusionPV_FeatureToken(
        hist_len=int(exp_conf["hist_len"]),
        pred_len=int(exp_conf["pred_len"]),
        var_num=int(exp_conf["var_num"]),
        pv_index=int(exp_conf.get("pv_index", 0)),
        token_dim=int(exp_conf.get("token_dim", 64)),
        layer_num=int(exp_conf.get("layer_num", 3)),
        layer_hp=exp_conf.get("layer_hp", None),
        use_layernorm=bool(exp_conf.get("use_layernorm", True)),
        dropout=float(exp_conf.get("dropout", 0.1)),
        baseline_mode=str(exp_conf.get("baseline_mode", "last24_repeat")),
        use_pv_revin=bool(exp_conf.get("use_pv_revin", True)),
        revin_eps=float(exp_conf.get("revin_eps", 1e-5)),
        enforce_nonneg=bool(exp_conf.get("enforce_nonneg", True)),
    )

    load_ckpt_to_model(model, args.ckpt)
    model.eval()
    model.to(device)

    # load x
    x_path = os.path.join(args.dataset_dir, f"{args.split}_x.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"not found: {x_path}")
    X = np.load(x_path)  # expected (N, 96, 9)
    if X.ndim != 3:
        raise ValueError(f"{x_path} shape unexpected: {X.shape}")

    N, L, F = X.shape
    print(f"[INFO] loaded X: {x_path}, shape={X.shape}")
    if args.window_index < 0 or args.window_index >= N:
        raise IndexError(f"window_index out of range: {args.window_index} (0..{N-1})")

    # 1サンプルを batch_size 分だけ複製して安定した平均を取る
    x0 = X[args.window_index]                      # (L, F)
    xb = np.repeat(x0[None, :, :], args.batch_size, axis=0)  # (B, L, F)
    var_x = torch.from_numpy(xb).float().to(device)

    # forward: gate を取得（gates: list[Tensor] 各 (B, F, E)） :contentReference[oaicite:5]{index=5}
    with torch.no_grad():
        pred, gates_all = model(var_x, marker_x=None, return_gate=True)

    # 保存
    os.makedirs(args.outdir, exist_ok=True)

    # 予測も一応保存（ヒートマップに直接は不要だが、同じ入力での動作確認に便利）
    pred_np = pred.detach().cpu().numpy()  # (B, 96, 1)
    np.save(os.path.join(args.outdir, "pred_sample.npy"), pred_np)

    # 各層の gate を (E,F) にして保存
    for layer_i, g in enumerate(gates_all):
        # g: (B, F, E)
        if not torch.is_tensor(g):
            raise TypeError(f"gate is not tensor at layer {layer_i}")
        if g.ndim != 3:
            raise ValueError(f"gate shape unexpected at layer {layer_i}: {tuple(g.shape)}")

        g_mean = g.float().mean(dim=0)       # (F, E)
        ev = g_mean.transpose(0, 1).cpu().numpy()  # (E, F)

        out_csv = os.path.join(args.outdir, f"layer{layer_i}_gate_EF.csv")
        out_png = os.path.join(args.outdir, f"layer{layer_i}_gate_EF.png")
        np.savetxt(out_csv, ev, delimiter=",")

        title = f"Layer {layer_i} gate mean (E x F)"
        save_heatmap(ev, out_png, title=title, feature_names=feature_names)

        print(f"[INFO] saved: {out_png}")

    # meta も一緒に書き出して再現性確保
    with open(os.path.join(args.outdir, "meta_used.txt"), "w", encoding="utf-8") as f:
        f.write(f"config={args.config}\n")
        f.write(f"ckpt={args.ckpt}\n")
        f.write(f"dataset_dir={args.dataset_dir}\n")
        f.write(f"split={args.split}\n")
        f.write(f"batch_size={args.batch_size}\n")
        f.write(f"window_index={args.window_index}\n")
        f.write("feature_names=" + ",".join(feature_names) + "\n")

    print("[INFO] done.")


if __name__ == "__main__":
    main()
