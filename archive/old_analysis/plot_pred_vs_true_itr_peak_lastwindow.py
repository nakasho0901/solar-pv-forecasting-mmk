# -*- coding: utf-8 -*-
r"""
最後の窓（テストデータの最後サンプル）に対して、
実測値（True）と、夜間補正した予測値（Pred）を重ねたグラフを保存するスクリプト。

夜間0補正（後処理）：
- 第一候補：真のGHI（var_y内）で夜間判定し、GHI <= threshold のとき Pred PV を 0 にする
- フォールバック：var_y に GHI が無い場合、真のPVが十分小さい（<= pv_eps）ところを夜間とみなし Pred PV を 0 にする

改善点（今回の要望）：
- Pred(raw) は描かない
- 凡例がグラフに被らないように「外側」に配置し、余白を確保する
"""

import os
import sys
import argparse
import importlib.util

import numpy as np
import torch
import matplotlib.pyplot as plt

# =========================
#  パス設定（プロジェクト直下想定）
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tools/ の1つ上
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from easytsf.runner.data_runner import NPYDataInterface
from traingraph import get_model, PeakRunner


def _get_last_batch_from_loader(loader):
    """DataLoader を最後まで回して「最後のバッチ」を返す。"""
    last_batch = None
    for batch in loader:
        last_batch = batch
    if last_batch is None:
        raise RuntimeError("[ERROR] test_dataloader からバッチが取得できませんでした。データが空の可能性があります。")
    return last_batch


def _load_config(config_path: str) -> dict:
    """configファイルから exp_conf を取り出す。"""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    if not hasattr(config_module, "exp_conf"):
        raise RuntimeError("[ERROR] config に exp_conf が見つかりません。")
    return config_module.exp_conf


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="configファイルのパス")
    parser.add_argument("--ckpt", type=str, required=True, help="Lightning checkpoint (.ckpt) のパス")
    parser.add_argument("--outdir", type=str, default="results_predplots_itr", help="出力フォルダ")
    parser.add_argument("--batch_size", type=int, default=256, help="test loader のバッチサイズ")
    parser.add_argument("--device", type=str, default="cpu", help="cpu か cuda (例: cuda:0)")

    # 夜間判定パラメータ
    parser.add_argument("--ghi_threshold", type=float, default=10.0,
                        help="GHI <= この値のとき夜間として Pred PV を0にする（GHIが取れた場合のみ）")
    parser.add_argument("--pv_eps", type=float, default=0.5,
                        help="(フォールバック) True PV <= この値を夜間とみなし Pred PV を0にする")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # -------------------------
    # config 読み込み
    # -------------------------
    exp_conf = _load_config(args.config)
    pv_index = int(exp_conf.get("pv_index", 0))
    ghi_index = int(exp_conf.get("ghi_index", 3))

    # -------------------------
    # DataModule 準備
    # -------------------------
    data_module = NPYDataInterface(**exp_conf)
    if hasattr(data_module, "setup"):
        try:
            data_module.setup(stage="test")
        except TypeError:
            data_module.setup()

    if not hasattr(data_module, "test_dataloader"):
        raise RuntimeError("[ERROR] NPYDataInterface に test_dataloader が見つかりません。")

    # batch_size 上書き（可能なら）
    try:
        if hasattr(data_module, "batch_size"):
            data_module.batch_size = args.batch_size
    except Exception:
        pass

    test_loader = data_module.test_dataloader()

    # -------------------------
    # モデル＋Runner 構築
    # -------------------------
    base_model = get_model(exp_conf["model_name"], exp_conf)
    runner = PeakRunner(base_model, exp_conf)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "state_dict" not in ckpt:
        raise RuntimeError("[ERROR] ckpt に state_dict がありません。Lightning checkpoint 形式か確認してください。")

    missing, unexpected = runner.load_state_dict(ckpt["state_dict"], strict=False)
    if len(missing) > 0:
        print("[WARN] missing keys:", missing)
    if len(unexpected) > 0:
        print("[WARN] unexpected keys:", unexpected)

    device = torch.device(args.device)
    runner.to(device)
    runner.eval()

    # -------------------------
    # 「最後の窓」を取り出す
    # -------------------------
    var_x, marker_x, var_y, marker_y = _get_last_batch_from_loader(test_loader)

    var_x_last = var_x[-1:].float().to(device)
    marker_x_last = marker_x[-1:].float().to(device)
    var_y_last = var_y[-1:].float().to(device)

    # -------------------------
    # 予測
    # -------------------------
    pred_pv = runner._predict(var_x_last, marker_x_last)  # (1, T, 1) を期待

    # True PV 抽出（var_yが多変量なら pv_index を使う）
    if var_y_last.ndim == 3 and var_y_last.shape[-1] > 1:
        true_pv = var_y_last[:, :, pv_index:pv_index + 1]
    else:
        true_pv = var_y_last

    pred_1d = pred_pv.squeeze(0).squeeze(-1).detach().cpu().numpy()
    true_1d = true_pv.squeeze(0).squeeze(-1).detach().cpu().numpy()

    # -------------------------
    # 夜間判定 → Predを0に（GHI優先、無ければTruePVで代用）
    # -------------------------
    night_mask = None
    night_mode = None

    ghi_1d = None
    if var_y_last.ndim == 3 and var_y_last.shape[-1] > ghi_index:
        ghi_1d = var_y_last[0, :, ghi_index].detach().cpu().numpy()

    if ghi_1d is not None:
        night_mask = (ghi_1d <= float(args.ghi_threshold))
        night_mode = f"GHI<= {args.ghi_threshold}"
    else:
        night_mask = (true_1d <= float(args.pv_eps))
        night_mode = f"TruePV<= {args.pv_eps} (fallback)"
        print("[WARN] var_y に GHI 列が無いので、夜間判定を True PV で代用します。")

    pred_1d_clipped = pred_1d.copy()
    pred_1d_clipped[night_mask] = 0.0

    # -------------------------
    # 描画して保存（rawは描かない／凡例は外側）
    # -------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(true_1d, label="True")
    ax.plot(pred_1d_clipped, label=f"Pred (night->0 : {night_mode})")

    ax.set_title("Pred vs True (Last Window)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("PV")

    # 凡例を右外に出す（被り防止）
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)

    # 右側に凡例分の余白を作る
    fig.subplots_adjust(right=0.75)

    out_path = os.path.join(args.outdir, "pred_vs_true_lastwindow_nightclip.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] saved: {out_path}")
    print(f"[INFO] night steps: {int(night_mask.sum())} / {int(night_mask.size)} ({night_mode})")


if __name__ == "__main__":
    main()
