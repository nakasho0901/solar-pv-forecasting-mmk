# -*- coding: utf-8 -*-
# ckpt から予測を生成して保存（LTSFRunner.forward(batch, batch_idx)想定／4要素バッチ対応）
# 出力:
#   save/<model>_<dataset>/<exp_hash>/seed_1/preds/test_preds.npy
#   save/<model>_<dataset>/<exp_hash>/seed_1/preds/test_y_from_loader.npy

import os, sys, argparse, importlib
import numpy as np
import torch
import lightning.pytorch as L
from lightning.pytorch.loggers import CSVLogger
from easytsf.runner.data_runner import DataInterface
from easytsf.runner.exp_runner import LTSFRunner
from easytsf.util import load_module_from_path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_config(exp_conf_path):
    exp_conf = load_module_from_path("exp_conf", exp_conf_path).exp_conf
    task_conf_module = importlib.import_module('config.base_conf.task')
    data_conf_module = importlib.import_module('config.base_conf.datasets')
    task_conf = task_conf_module.task_conf
    data_conf = eval('data_conf_module.{}_conf'.format(exp_conf['dataset_name']))
    fused = {**task_conf, **data_conf}
    fused.update(exp_conf)
    return fused

def autodetect_data_root(dataset_name="Tsukuba"):
    for root in ["dataset", "dataset_PV"]:
        if os.path.exists(os.path.join(root, f"{dataset_name}.npz")):
            return root
    return "dataset"

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(t, device) for t in obj)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    return obj

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", required=True)
    ap.add_argument("--exp_hash", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    conf = load_config(args.config)
    conf["seed"] = args.seed
    L.seed_everything(conf["seed"])

    if "data_root" not in conf or not conf["data_root"]:
        conf["data_root"] = autodetect_data_root(conf.get("dataset_name","Tsukuba"))
    if "save_root" not in conf or not conf["save_root"]:
        conf["save_root"] = "save"

    save_dir = os.path.join(conf["save_root"], f'{conf["model_name"]}_{conf["dataset_name"]}')
    exp_dir  = os.path.join(save_dir, args.exp_hash, f'seed_{conf["seed"]}')
    preds_dir = os.path.join(exp_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)
    conf["exp_dir"] = exp_dir

    accelerator, devices = ("gpu", 1) if torch.cuda.is_available() else ("cpu", 1)
    logger = CSVLogger(save_dir=save_dir, name=args.exp_hash, version=f'seed_{conf["seed"]}')
    _ = L.Trainer(accelerator=accelerator, devices=devices, logger=logger, enable_checkpointing=False)

    data_module = DataInterface(**conf)
    model = LTSFRunner.load_from_checkpoint(args.ckpt)
    model.eval()

    dl = data_module.test_dataloader()
    preds_list, ys_list = [], []

    pred_len = int(conf.get("pred_len", 96))

    with torch.no_grad():
        for bidx, batch in enumerate(dl):
            # バッチは tuple/list で来る（想定: var_x, marker_x, var_y, marker_y）
            batch_on_dev = move_to_device(batch, model.device)

            # y を取り出す（第3要素が var_y のはず。なければ第2要素をfallback）
            if isinstance(batch, (list, tuple)):
                y = batch[2] if len(batch) >= 3 else (batch[1] if len(batch) >= 2 else None)
            elif isinstance(batch, dict):
                y = batch.get("var_y") or batch.get("y") or batch.get("labels")
            else:
                y = None
            if y is not None:
                ys_list.append(to_numpy(y))

            # forward は (batch, batch_idx) を要求
            out = model.forward(batch_on_dev, bidx)

            # 返り値がタプル/リストなら、(B, pred_len[,1]) 形状のものを優先選択
            if isinstance(out, (list, tuple)):
                cand = None
                for o in out:
                    o_np = to_numpy(o)
                    if o_np.ndim >= 2 and o_np.shape[1] == pred_len:
                        cand = o_np; break
                out_np = cand if cand is not None else to_numpy(out[0])
            else:
                out_np = to_numpy(out)

            # (B, pred_len, 1) → (B, pred_len)
            if out_np.ndim == 3 and out_np.shape[-1] == 1:
                out_np = out_np[..., 0]

            preds_list.append(out_np)

    preds = np.concatenate(preds_list, axis=0)
    if ys_list:
        Y = np.concatenate(ys_list, axis=0)
        if Y.ndim == 3 and Y.shape[-1] == 1:
            Y = Y[..., 0]
        assert preds.shape[:2] == Y.shape[:2], f"shape mismatch: preds{preds.shape} vs Y{Y.shape}"
        np.save(os.path.join(preds_dir, "test_y_from_loader.npy"), Y.astype(np.float32))
        print("Saved:", os.path.join(preds_dir, "test_y_from_loader.npy"))

    np.save(os.path.join(preds_dir, "test_preds.npy"), preds.astype(np.float32))
    print("Saved:", os.path.join(preds_dir, "test_preds.npy"))
