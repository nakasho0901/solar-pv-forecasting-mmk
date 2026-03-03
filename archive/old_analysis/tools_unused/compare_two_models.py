# -*- coding: utf-8 -*-
"""
2モデル（MMK / iTransformer）を同一サンプルで比較描画（96h）。
- DataLoaderは使わず test_x.npy/test_y.npy を直接 index して安全に取得
- iTransformer の marker_x 要求を自動吸収（0→1線形 or sin/cos）
- 出力shapeが (B, pred_len, C) のときはチャネル0を採用
- 追加オプション: --calibrate で a*pred+b の線形キャリブ（疑似逆変換）
使い方（例・1行）:
python tools/compare_two_models.py -c1 config/tsukuba_conf/MMK_Tsukuba_96for96_weather.py -c2 config/tsukuba_conf/iTransformer_Tsukuba_96for96_weather.py --global_idx -1 --hours 96 --out_dir figs
"""

import os, sys, glob, importlib.util, inspect, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---- import path ----
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---- helpers ----
def load_exp_conf(conf_path: str):
    spec = importlib.util.spec_from_file_location("exp_conf_mod", conf_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.exp_conf

def latest_ckpt(save_root: str, model_name: str):
    patt = os.path.join(save_root, f"{model_name}_Tsukuba", "**", "seed_1", "checkpoints", "*.ckpt")
    cands = glob.glob(patt, recursive=True)
    if not cands:
        patt = os.path.join(save_root, f"{model_name}*", "**", "checkpoints", "*.ckpt")
        cands = glob.glob(patt, recursive=True)
    return sorted(cands)[-1] if cands else None

def filter_and_bridge_args(Model, conf: dict) -> dict:
    """Model.__init__ の実引数だけ／必要なリネーム（seq_len→hist_len, enc_in→in_dim, c_out→out_dim 等）"""
    sig = inspect.signature(Model.__init__)
    need = {n for n,p in sig.parameters.items() if n != "self"}
    kw = {k:v for k,v in conf.items() if k in need}
    bridges = [("seq_len","hist_len"), ("enc_in","in_dim"), ("c_out","out_dim"), ("pred_len","pred_len"), ("dec_in","dec_in")]
    for src,dst in bridges:
        if dst in need and dst not in kw and src in conf:
            kw[dst] = conf[src]
    if "var_num" in need and "var_num" not in kw:
        kw["var_num"] = 1
    return kw

def make_time_marker(xb: torch.Tensor) -> torch.Tensor:
    """(B,L,1) 0→1 の線形時間マーカー"""
    B,L = xb.size(0), xb.size(1)
    t = torch.linspace(0.0, 1.0, L, device=xb.device).view(1,L,1)
    return t.expand(B,L,1).contiguous()

def make_time_marker_sincos(xb: torch.Tensor) -> torch.Tensor:
    """(B,L,2) sin/cos 時間埋め込み"""
    B,L = xb.size(0), xb.size(1)
    tt = torch.linspace(0.0, 1.0, L, device=xb.device) * 2 * torch.pi
    sin = torch.sin(tt).view(1,L,1)
    cos = torch.cos(tt).view(1,L,1)
    mk = torch.cat([sin, cos], dim=-1)
    return mk.expand(B,L,2).contiguous()

def forward_with_marker(model, xb: torch.Tensor):
    """forward 引数差を吸収"""
    try:
        return model(xb)
    except TypeError:
        pass
    mk = make_time_marker(xb)
    try:
        return model(xb, mk)
    except TypeError:
        try:
            return model(xb, marker_x=mk)
        except TypeError:
            mk2 = make_time_marker_sincos(xb)
            try:
                return model(xb, mk2)
            except TypeError:
                return model(xb, marker_x=mk2)

def to_2d_pred(yb: torch.Tensor) -> torch.Tensor:
    """(B, pred_len[, C]) → (B, pred_len)"""
    if yb.dim() == 3:
        return yb.squeeze(-1) if yb.size(-1) == 1 else yb[...,0]
    return yb

def mae(a, b): return float(np.mean(np.abs(a-b)))
def mse(a, b): return float(np.mean((a-b)**2))

def calibrate_to_y(pred: np.ndarray, y: np.ndarray):
    """最小二乗で a*pred + b をあてて実スケールへ擬似逆変換"""
    A = np.c_[pred.reshape(-1), np.ones(pred.size)]
    a,b = np.linalg.lstsq(A, y.reshape(-1), rcond=None)[0]
    return a*pred + b, float(a), float(b)

# ---- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c1","--config1", required=True)
    ap.add_argument("-c2","--config2", required=True)
    ap.add_argument("--global_idx", type=int, default=-1, help="-1=最後のサンプル")
    ap.add_argument("--hours", type=int, default=96)
    ap.add_argument("--out_dir", default="figs")
    ap.add_argument("--calibrate", action="store_true", help="a*pred+b で擬似逆変換してから描画")
    args = ap.parse_args()

    # 構成読み込み
    conf1 = load_exp_conf(args.config1)
    conf2 = load_exp_conf(args.config2)
    name1 = conf1.get("model_name", conf1.get("model","Model1"))
    name2 = conf2.get("model_name", conf2.get("model","Model2"))
    data_path = conf1["data_path"]
    assert data_path == conf2["data_path"], "両モデルの data_path が一致していません"

    # データ読込（直接 index）
    Y = np.load(os.path.join(data_path, "test_y.npy"))       # (N, pred_len)
    X = np.load(os.path.join(data_path, "test_x.npy"))       # (N, L, C)
    N, L, C = X.shape
    idx = (N-1) if args.global_idx == -1 else args.global_idx
    if not (0 <= idx < N):
        raise IndexError(f"global_idx out of range: {idx}/{N}")

    # モデル1 構築・読み込み
    if name1.lower().startswith("itransformer"):
        from easytsf.model.iTransformer import iTransformer as Model1
    elif name1.lower().startswith("mmk"):
        from easytsf.model.MMK import MMK as Model1
    else:
        raise RuntimeError(f"unknown model1: {name1}")
    kw1 = filter_and_bridge_args(Model1, conf1)
    m1 = Model1(**kw1).to("cpu").eval()
    ckpt1 = latest_ckpt(conf1.get("save_root","save"), name1)
    if ckpt1 is None: raise FileNotFoundError("ckpt for model1 not found")
    sd1 = torch.load(ckpt1, map_location="cpu")
    sd1 = sd1.get("state_dict", sd1)
    try: m1.load_state_dict(sd1, strict=False)
    except Exception:
        m1.load_state_dict({k.split("model.",1)[-1]:v for k,v in sd1.items()}, strict=False)

    # モデル2
    if name2.lower().startswith("itransformer"):
        from easytsf.model.iTransformer import iTransformer as Model2
    elif name2.lower().startswith("mmk"):
        from easytsf.model.MMK import MMK as Model2
    else:
        raise RuntimeError(f"unknown model2: {name2}")
    kw2 = filter_and_bridge_args(Model2, conf2)
    m2 = Model2(**kw2).to("cpu").eval()
    ckpt2 = latest_ckpt(conf2.get("save_root","save"), name2)
    if ckpt2 is None: raise FileNotFoundError("ckpt for model2 not found")
    sd2 = torch.load(ckpt2, map_location="cpu")
    sd2 = sd2.get("state_dict", sd2)
    try: m2.load_state_dict(sd2, strict=False)
    except Exception:
        m2.load_state_dict({k.split("model.",1)[-1]:v for k,v in sd2.items()}, strict=False)

    # 単一サンプルをテンソル化（B=1）
    xb = torch.from_numpy(X[idx:idx+1]).float()   # (1,L,C)
    gt = Y[idx]                                   # (hours)

    # 推論
    with torch.no_grad():
        y1 = to_2d_pred(forward_with_marker(m1, xb)).cpu().numpy()[0]
        y2 = to_2d_pred(forward_with_marker(m2, xb)).cpu().numpy()[0]

    # 必要ならキャリブ（疑似逆変換）
    if args.calibrate:
        y1_cal, a1, b1 = calibrate_to_y(y1, gt)
        y2_cal, a2, b2 = calibrate_to_y(y2, gt)
        y1, y2 = y1_cal, y2_cal
        ab_text = f" | cal: a1={a1:.3f},b1={b1:.3f}; a2={a2:.3f},b2={b2:.3f}"
    else:
        ab_text = ""

    # 部分表示（hours で頭から or 末尾96? → 予測は基本96長なので全量表示）
    H = min(args.hours, gt.shape[0])
    x = np.arange(H)
    gt_v = gt[:H]
    y1_v = y1[:H]
    y2_v = y2[:H]

    # サンプルMAE/MSE
    mae1, mse1 = mae(y1_v, gt_v), mse(y1_v, gt_v)
    mae2, mse2 = mae(y2_v, gt_v), mse(y2_v, gt_v)

    # 描画
    os.makedirs(args.out_dir, exist_ok=True)
    plt.figure(figsize=(12,4))
    plt.plot(x, gt_v, label="True", linewidth=2.0)
    plt.plot(x, y1_v, "--", label=f"{name1}  MAE={mae1:.2f}, MSE={mse1:.1f}")
    plt.plot(x, y2_v, "--", label=f"{name2}  MAE={mae2:.2f}, MSE={mse2:.1f}")
    plt.title(f"Compare @idx={idx}{ab_text}")
    plt.xlabel("Hour"); plt.ylabel("Value")
    plt.legend(); plt.tight_layout()
    out_path = os.path.join(args.out_dir, f"compare_{name1}_vs_{name2}_idx{idx}.png")
    plt.savefig(out_path, dpi=200)
    print(f"[OK] saved: {out_path}")

if __name__ == "__main__":
    main()
