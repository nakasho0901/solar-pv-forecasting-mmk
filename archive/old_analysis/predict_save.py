# -*- coding: utf-8 -*-
"""
学習済み checkpoint を使って test 予測を保存するスクリプト（完全版）
- iTransformer / MMK を切り替え可能
- hardzero config も切替可能
- --config で任意config指定も可能
- save配下の seed_x/version_y/checkpoints から「最新version」＋「最新ckpt」を自動で選ぶ
- exp_runner.forward が (pred, label, marker_y) を返す版に対応

使い方例:
  # 通常
  python predict_save.py --model itr
  python predict_save.py --model mmk

  # hardzero
  python predict_save.py --model itr_hardzero
  python predict_save.py --model mmk_hardzero

  # 任意config指定（推奨）
  python predict_save.py --config config\\tsukuba_conf\\iTransformer_PVProcessed_96for96_hardzero.py --seed 2025
"""

import argparse
import importlib.util
import os
from typing import Dict, Optional, Tuple, List

import numpy as np
import torch

from easytsf.runner.data_runner import NPYDataInterface as DataInterface
from easytsf.runner.exp_runner import LTSFRunner


# -----------------------------
# config loader
# -----------------------------
def load_config(config_path: str) -> Dict:
    module_name = os.path.splitext(os.path.basename(config_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"config を読み込めません: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "exp_conf"):
        raise AttributeError(f"config に exp_conf がありません: {config_path}")
    return module.exp_conf


# -----------------------------
# checkpoint finder
# -----------------------------
def list_version_dirs(model_root: str) -> List[Tuple[int, str]]:
    """model_root 以下の version_x を全部列挙して (num, dirname) を返す"""
    versions = []
    if not os.path.isdir(model_root):
        return versions
    for name in os.listdir(model_root):
        if name.startswith("version_"):
            try:
                num = int(name.split("_")[1])
                versions.append((num, name))
            except Exception:
                pass
    versions.sort(key=lambda x: x[0])
    return versions


def find_latest_version_dir(model_root: str) -> str:
    versions = list_version_dirs(model_root)
    if not versions:
        raise FileNotFoundError("version_x が見つかりません: " + model_root)
    return versions[-1][1]


def pick_latest_ckpt(ckpt_dir: str) -> str:
    """ckpt_dir 内の .ckpt を更新時刻でソートして最新を返す"""
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError("checkpoints ディレクトリが見つかりません: " + ckpt_dir)

    ckpts = []
    for f in os.listdir(ckpt_dir):
        if f.endswith(".ckpt"):
            full = os.path.join(ckpt_dir, f)
            try:
                mtime = os.path.getmtime(full)
            except Exception:
                mtime = 0.0
            ckpts.append((mtime, full))

    if not ckpts:
        raise FileNotFoundError("checkpoint (.ckpt) が見つかりません: " + ckpt_dir)

    ckpts.sort(key=lambda x: x[0])
    return ckpts[-1][1]


def resolve_config_from_model_key(model_key: str) -> str:
    """--model の短縮キーから config パスに変換"""
    mapping = {
        "itr": "config/tsukuba_conf/iTransformer_PVProcessed_96for96.py",
        "mmk": "config/tsukuba_conf/MMK_PVProcessed_96for96.py",
        "itr_hardzero": "config/tsukuba_conf/iTransformer_PVProcessed_96for96_hardzero.py",
        "mmk_hardzero": "config/tsukuba_conf/MMK_PVProcessed_96for96_hardzero.py",
    }
    if model_key not in mapping:
        raise ValueError(
            f"--model は {list(mapping.keys())} のいずれか、または --config を指定してください。"
        )
    return mapping[model_key]


def infer_model_root(save_root: str, config_path: str, seed: int) -> str:
    """
    traingraph.py の保存規約に合わせる:
      save/<config_basename>/seed_<seed>/version_x/checkpoints/...
    """
    config_base = os.path.splitext(os.path.basename(config_path))[0]
    return os.path.join(save_root, config_base, f"seed_{seed}")


# -----------------------------
# main
# -----------------------------
def main(config_path: str, save_root: str, seed: int, out_dir: Optional[str]) -> str:
    print(f"Using config: {config_path}")
    conf = load_config(config_path)

    # DataLoader (test)
    data_module = DataInterface(**conf)
    data_module.setup("test")
    test_loader = data_module.test_dataloader()

    # checkpoint auto detect
    model_root = infer_model_root(save_root, config_path, seed)
    version_dir = find_latest_version_dir(model_root)
    ckpt_dir = os.path.join(model_root, version_dir, "checkpoints")
    ckpt_path = pick_latest_ckpt(ckpt_dir)

    print("Using model_root :", model_root)
    print("Using version   :", version_dir)
    print("Using checkpoint:", ckpt_path)

    # load model
    model = LTSFRunner.load_from_checkpoint(ckpt_path, **conf)
    model.eval()

    preds = []
    trues = []

    # forward は (pred, label, marker_y) or (pred, label) の両対応にする
    with torch.no_grad():
        for batch in test_loader:
            out = model.forward(batch, 0)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                y_pred, y_true = out[0], out[1]
            else:
                raise RuntimeError("model.forward の戻り値が想定外です（tuple/listで2要素以上が必要）")

            preds.append(y_pred.detach().cpu().numpy())
            trues.append(y_true.detach().cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # 保存先
    # 既存のあなたの運用（.../seed_2025/preds）に合わせる
    if out_dir is None:
        out_dir = os.path.join(model_root, "preds")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "test_preds.npy"), preds)
    np.save(os.path.join(out_dir, "true_test.npy"), trues)

    print("Saved prediction files in:", out_dir)
    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        type=str,
        help="itr / mmk / itr_hardzero / mmk_hardzero のいずれか",
    )
    group.add_argument(
        "--config",
        type=str,
        help="任意の config ファイルを指定（こちらが最も確実）",
    )

    parser.add_argument("--save_root", type=str, default="save", help="checkpoint のルートディレクトリ")
    parser.add_argument("--seed", type=int, default=2025, help="seed番号（save/..../seed_<seed>）")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="予測保存先（指定しない場合は save/<config>/seed_<seed>/preds）",
    )

    args = parser.parse_args()

    if args.config:
        config_path = args.config
    else:
        config_path = resolve_config_from_model_key(args.model)

    main(config_path=config_path, save_root=args.save_root, seed=args.seed, out_dir=args.out_dir)
