# -*- coding: utf-8 -*-
# 学習済み ckpt から test 予測を保存: preds/test_preds.npy
# 改善点:
#  - --ckpt で任意の ckpt を明示指定できる
#  - 自動探索時は「更新時刻(mtime)の新しい順」で選ぶ（文字列ソートを排除）
#  - iTransformer の marker_x を時間埋め込みで自動付与
#  - 多チャネル出力はチャネル0(PV)に圧縮

import os, glob, importlib.util, sys, inspect, argparse, time
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def load_exp_conf(conf_path: str):
    spec = importlib.util.spec_from_file_location("exp_conf_mod", conf_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.exp_conf

def latest_ckpt_by_mtime(save_root: str, model_name: str):
    patt1 = os.path.join(save_root, f"{model_name}_Tsukuba", "**", "checkpoints", "*.ckpt")
    patt2 = os.path.join(save_root, f"{model_name}*", "**", "checkpoints", "*.ckpt")
    cands = glob.glob(patt1, recursive=True) + glob.glob(patt2, recursive=True)
    if not cands:
        return None
    # mtime（更新時刻）の新しい順に並べて先頭を返す
    cands.sort(key=lambda p: os.path.getmtime(p))
    return cands[-1]

def filter_and_bridge_args(Model, conf: dict) -> dict:
    sig = inspect.signature(Model.__init__)
    need = {name for name, p in sig.parameters.items() if name != "self"}
    kw = {k: v for k, v in conf.items() if k in need}
    bridges = [("seq_len","hist_len"), ("enc_in","in_dim"), ("c_out","out_dim"), ("pred_len","pred_len"), ("dec_in","dec_in")]
    for src, dst in bridges:
        if dst in need and dst not in kw and src in conf:
            kw[dst] = conf[src]
    if "var_num" in need and "var_num" not in kw:
        kw["var_num"] = 1
    return kw

def make_time_marker(xb: torch.Tensor) -> torch.Tensor:
    B, L = xb.size(0), xb.size(1)
    t = torch.linspace(0.0, 1.0, L, device=xb.device).view(1, L, 1)
    return t.expand(B, L, 1).contiguous()

def make_time_marker_sincos(xb: torch.Tensor) -> torch.Tensor:
    B, L = xb.size(0), xb.size(1)
    tt = torch.linspace(0.0, 1.0, L, device=xb.device) * 2 * torch.pi
    sin = torch.sin(tt).view(1, L, 1)
    cos = torch.cos(tt).view(1, L, 1)
    mk = torch.cat([sin, cos], dim=-1)
    return mk.expand(B, L, 2).contiguous()

def forward_with_marker(model, xb: torch.Tensor):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", required=True)
    ap.add_argument("--save_dir", default=None, help="出力先（未指定なら save/<model>_Tsukuba/preds）")
    ap.add_argument("--ckpt", default=None, help="明示的に使う ckpt のフルパス")
    args = ap.parse_args()

    conf = load_exp_conf(args.config)
    model_name = conf.get("model_name", conf.get("model", "Model"))
    pred_len   = int(conf.get("pred_len", 96))
    data_path  = conf["data_path"]
    save_root  = conf.get("save_root","save")

    tx = np.load(os.path.join(data_path, "test_x.npy"))  # (N, L, C)
    device = torch.device("cpu")
    x = torch.from_numpy(tx).float().to(device)

    if model_name.lower().startswith("itransformer"):
        from easytsf.model.iTransformer import iTransformer as Model
    elif model_name.lower().startswith("mmk"):
        from easytsf.model.MMK import MMK as Model
    else:
        raise RuntimeError(f"unknown model_name: {model_name}")

    kw = filter_and_bridge_args(Model, conf)
    model = Model(**kw).to(device)
    model.eval()

    ckpt = args.ckpt or latest_ckpt_by_mtime(save_root, model_name)
    if ckpt is None:
        raise FileNotFoundError("ckpt not found under save/.")
    state = torch.load(ckpt, map_location=device)
    sd = state.get("state_dict", state)
    try:
        model.load_state_dict(sd, strict=False)
    except Exception:
        new_sd = {k.split("model.",1)[-1]: v for k,v in sd.items()}
        model.load_state_dict(new_sd, strict=False)

    preds, bs = [], 256
    with torch.no_grad():
        for i in range(0, x.size(0), bs):
            xb = x[i:i+bs]
            yb = forward_with_marker(model, xb)
            if yb.dim() == 3:
                yb = yb.squeeze(-1) if yb.size(-1) == 1 else yb[..., 0]
            preds.append(yb.cpu().numpy())

    P = np.concatenate(preds, axis=0)  # (N, pred_len)
    if P.ndim != 2:
        raise RuntimeError(f"unexpected pred shape: {P.shape}")
    if P.shape[1] != pred_len:
        print(f"[WARN] pred_len mismatch: {P.shape[1]} vs conf pred_len {pred_len}")

    out_dir = args.save_dir or os.path.join(save_root, f"{model_name}_Tsukuba", "preds")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test_preds.npy")
    np.save(out_path, P)
    mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(ckpt)))
    print(f"[OK] saved preds: {out_path} shape={P.shape} | ckpt={ckpt} (mtime={mtime})")

if __name__ == "__main__":
    main()
