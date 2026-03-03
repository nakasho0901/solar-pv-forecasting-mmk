# train.py
# -*- coding: utf-8 -*-

import argparse
import importlib
import os

import lightning.pytorch as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    EarlyStopping,
)
from lightning.pytorch.loggers import CSVLogger
import torch

# NPYDataInterface を DataInterface として使う
from easytsf.runner.data_runner import NPYDataInterface as DataInterface
from easytsf.runner.exp_runner import LTSFRunner


def load_config(config_path: str):
    """
    設定ファイル（.py）を読み込み、exp_conf を取得する
    """
    module_name = os.path.splitext(os.path.basename(config_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.exp_conf


def train_func(training_conf: dict, init_exp_conf: dict):
    """
    学習パイプライン本体
    """
    # 📌 Lightning seed
    if "seed" in init_exp_conf:
        L.seed_everything(init_exp_conf["seed"], workers=True)

    # 📌 DataModule の作成
    print("[INFO] NPYDataInterface を使用します（窓切り済み .npy データセット）")
    data_module = DataInterface(**training_conf)

    # 📌 モデルの作成
    model = LTSFRunner(**training_conf)

    # 📌 ロガー設定（sub_dir は v2 では非推奨のため name に統合）
    logger = CSVLogger(
        save_dir=init_exp_conf["save_dir"],
        name=f"{init_exp_conf['exp_name']}/seed_{init_exp_conf['seed']}",
    )

    # 📌 コールバック
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    ckpt_callback = ModelCheckpoint(
        monitor=training_conf["val_metric"],
        save_top_k=1,
        mode="min",
        filename="epoch={epoch}-step={step}",
    )

    early_stop = EarlyStopping(
        monitor=training_conf["val_metric"],
        patience=training_conf.get("early_stop_patience", 5),
        verbose=True,
        mode="min",
    )

    # 📌 Trainer
    trainer = L.Trainer(
        max_epochs=training_conf["max_epochs"],
        logger=logger,
        callbacks=[lr_monitor, ckpt_callback, early_stop],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=init_exp_conf.get("devices", 1),
        num_sanity_val_steps=2,
        log_every_n_steps=10,
    )

    # 📌 学習
    trainer.fit(model=model, datamodule=data_module)

    # 📌 テスト
    trainer.test(model=model, datamodule=data_module)


def parse_args():
    parser = argparse.ArgumentParser(description="Train LTSF Model (iTransformer + PVProcessed .npy)")
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-s", "--save_dir", type=str, required=True)
    parser.add_argument("-d", "--data_dir", type=str, default=".")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 📌 config 読み込み
    training_conf = load_config(args.config)

    # （必要ならここで data_dir を上書きしてもよいが、
    #  今回は config 側で data_dir="dataset_PV/processed" と固定している）

    # 📌 追加パラメータ
    init_exp_conf = {
        "save_dir": args.save_dir,
        "exp_name": os.path.splitext(os.path.basename(args.config))[0],
        "devices": args.devices,
        "seed": args.seed,
    }

    # 📌 実行
    train_func(training_conf, init_exp_conf)
