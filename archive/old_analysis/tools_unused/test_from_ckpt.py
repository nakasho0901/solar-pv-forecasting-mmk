# tools/test_from_ckpt.py
# ckpt 内のハイパラで LTSFRunner を復元し、テストのみ実行して preds/test_preds.npy を生成

import os, sys, argparse, importlib
import lightning.pytorch as L
from lightning.pytorch.loggers import CSVLogger
from easytsf.runner.data_runner import DataInterface
from easytsf.runner.exp_runner import LTSFRunner
from easytsf.util import load_module_from_path

# プロジェクト直下を import パスに追加（保険）
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", required=True)
    ap.add_argument("--exp_hash", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    # 1) データ周りの conf（モデルのハイパラは ckpt に任せる）
    conf = load_config(args.config)
    conf["seed"] = args.seed
    L.seed_everything(conf["seed"])

    if "data_root" not in conf or not conf["data_root"]:
        conf["data_root"] = autodetect_data_root(conf.get("dataset_name","Tsukuba"))
    if "save_root" not in conf or not conf["save_root"]:
        conf["save_root"] = "save"

    save_dir = os.path.join(conf["save_root"], f'{conf["model_name"]}_{conf["dataset_name"]}')
    exp_dir  = os.path.join(save_dir, args.exp_hash, f'seed_{conf["seed"]}')
    os.makedirs(exp_dir, exist_ok=True)
    conf["exp_dir"] = exp_dir

    # 2) デバイス自動判定
    try:
        import torch
        accelerator, devices = ("gpu", 1) if torch.cuda.is_available() else ("cpu", 1)
    except Exception:
        accelerator, devices = ("cpu", 1)

    logger = CSVLogger(save_dir=save_dir, name=args.exp_hash, version=f'seed_{conf["seed"]}')
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=conf.get("precision","32-true"),
        logger=logger,
    )

    # 3) DataInterface は conf を使用（入出力次元・data_path 用）
    data_module = DataInterface(**conf)

    # 4) モデルは ckpt から **そのまま** 復元（学習時ハイパラを使用）
    model = LTSFRunner.load_from_checkpoint(args.ckpt)

    # 5) テストのみ実行（ckpt_path=None で OK）
    trainer.test(model=model, datamodule=data_module, ckpt_path=None)

    print("Done. preds saved under:", os.path.join(exp_dir, "preds"))
