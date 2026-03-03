# -*- coding: utf-8 -*-
"""
traingraph_FusionPV.py

目的:
- NPYDataInterface に依存せず、data_dir 配下の .npy を直接読み込んで学習する（環境差で壊れない）。
- model_name="MMK_Mix_FusionPV" を含むモデル分岐を持つ。
- marker_x / marker_y が存在しない場合でも「サンプル数Bが揃うゼロテンソル」を自動生成し、
  TensorDataset の Size mismatch を確実に回避する（今回のエラーを一発で解消）。

実行例（CMD 1行）:
python traingraph_FusionPV.py -c config\\tsukuba_conf\\MMK_Mix_FusionPV_RevIN_PV_96.py -s save --devices 1 --seed 2025
"""

import os
import sys
import argparse
import importlib
import inspect
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from lightning.pytorch import Trainer, seed_everything, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger


# -------------------------
# Utils
# -------------------------
def _load_npy(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] file not found: {path}")
    return np.load(path)


def _ensure_3d_last1(t: torch.Tensor) -> torch.Tensor:
    """
    (B,T) -> (B,T,1)
    (B,T,1) -> keep
    """
    if t.dim() == 2:
        return t.unsqueeze(-1)
    if t.dim() == 3:
        return t
    raise ValueError(f"[ERROR] tensor must be 2D/3D, got {tuple(t.shape)}")


def _ensure_marker_3d(marker: torch.Tensor, length_expected: int) -> torch.Tensor:
    """
    marker の形をなるべく吸収する。
    - (B,L) -> (B,L,1)
    - (B,L,C) -> keep
    長さLが違う場合は警告のみ（必要ならここで調整）
    """
    m = _ensure_3d_last1(marker)
    if m.shape[1] != length_expected:
        print(f"[WARN] marker length mismatch: marker.shape[1]={m.shape[1]} vs expected={length_expected}")
    return m


def _extract_pred_to_3d1(outputs, pv_index_if_multi: int = 0) -> torch.Tensor:
    """
    モデル出力から PV 1系列の (B,T,1) を取り出す（互換重視）
    - outputs が tuple/list の場合: outputs[0] を pred とみなす
    - pred が (B,T,1): そのまま
    - pred が (B,T,C): pv_index_if_multi で 1本にする
    - pred が (B,T): (B,T,1) にする
    """
    pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

    if pred.dim() == 2:
        return pred.unsqueeze(-1)
    if pred.dim() == 3 and pred.shape[-1] == 1:
        return pred
    if pred.dim() == 3 and pred.shape[-1] > 1:
        idx = pv_index_if_multi if pv_index_if_multi < pred.shape[-1] else 0
        return pred[:, :, idx:idx + 1]

    raise ValueError(f"[ERROR] pred shape unexpected: {tuple(pred.shape)}")


# -------------------------
# DataModule (NPY direct)
# -------------------------
class NPYWindowDataModule:
    """
    .npy を直接読むシンプル DataModule。
    marker が無い場合でも TensorDataset の B を揃えるゼロテンソルを作る。
    """
    def __init__(self, exp_conf: Dict):
        self.conf = exp_conf
        self.data_dir = exp_conf["data_dir"]

        self.hist_len = int(exp_conf.get("hist_len", exp_conf.get("seq_len", 96)))
        self.pred_len = int(exp_conf.get("pred_len", exp_conf.get("fut_len", 96)))

        self.batch_size = int(exp_conf.get("batch_size", 64))
        self.num_workers = int(exp_conf.get("num_workers", 0))
        self.pin_memory = bool(exp_conf.get("pin_memory", False))

        # filenames
        self.train_x = exp_conf.get("train_x", "train_x.npy")
        self.train_y = exp_conf.get("train_y", "train_y.npy")
        self.val_x = exp_conf.get("val_x", "val_x.npy")
        self.val_y = exp_conf.get("val_y", "val_y.npy")
        self.test_x = exp_conf.get("test_x", "test_x.npy")
        self.test_y = exp_conf.get("test_y", "test_y.npy")

        # optional marker
        self.train_marker_x = exp_conf.get("train_marker_x", "train_marker_x.npy")
        self.val_marker_x = exp_conf.get("val_marker_x", "val_marker_x.npy")
        self.test_marker_x = exp_conf.get("test_marker_x", "test_marker_x.npy")

        self.train_marker_y = exp_conf.get("train_marker_y", "train_marker_y.npy")
        self.val_marker_y = exp_conf.get("val_marker_y", "val_marker_y.npy")
        self.test_marker_y = exp_conf.get("test_marker_y", "test_marker_y.npy")

        self._train_ds = None
        self._val_ds = None
        self._test_ds = None

    def _paths_for(self, split: str) -> Tuple[str, str, str, str]:
        if split == "train":
            return (
                os.path.join(self.data_dir, self.train_x),
                os.path.join(self.data_dir, self.train_y),
                os.path.join(self.data_dir, self.train_marker_x),
                os.path.join(self.data_dir, self.train_marker_y),
            )
        if split == "val":
            return (
                os.path.join(self.data_dir, self.val_x),
                os.path.join(self.data_dir, self.val_y),
                os.path.join(self.data_dir, self.val_marker_x),
                os.path.join(self.data_dir, self.val_marker_y),
            )
        if split == "test":
            return (
                os.path.join(self.data_dir, self.test_x),
                os.path.join(self.data_dir, self.test_y),
                os.path.join(self.data_dir, self.test_marker_x),
                os.path.join(self.data_dir, self.test_marker_y),
            )
        raise ValueError(split)

    def _load_split(self, split: str) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        x_path, y_path, mx_path, my_path = self._paths_for(split)

        x = torch.from_numpy(_load_npy(x_path)).float()  # (B,L,N)
        y = torch.from_numpy(_load_npy(y_path)).float()  # (B,T) or (B,T,1)
        y = _ensure_3d_last1(y)  # (B,T,1)

        if x.dim() != 3:
            raise ValueError(f"[ERROR] {split}_x must be 3D (B,L,N), got {tuple(x.shape)}")

        if x.shape[1] != self.hist_len:
            print(f"[WARN] hist_len mismatch: x.shape[1]={x.shape[1]} vs conf hist_len={self.hist_len}")

        if y.shape[1] != self.pred_len:
            print(f"[WARN] pred_len mismatch: y.shape[1]={y.shape[1]} vs conf pred_len={self.pred_len}")

        mx = None
        if os.path.exists(mx_path):
            mx = torch.from_numpy(_load_npy(mx_path)).float()
            mx = _ensure_marker_3d(mx, length_expected=x.shape[1])

        my = None
        if os.path.exists(my_path):
            my = torch.from_numpy(_load_npy(my_path)).float()
            my = _ensure_marker_3d(my, length_expected=y.shape[1])

        return x, y, mx, my

    def setup(self):
        # ---- train ----
        x, y, mx, my = self._load_split("train")
        B, L, _ = x.shape
        T = y.shape[1]
        if mx is None:
            mx = torch.zeros((B, L, 1), dtype=torch.float32)
        if my is None:
            my = torch.zeros((B, T, 1), dtype=torch.float32)
        self._train_ds = TensorDataset(x, mx, y, my)

        # ---- val ----
        x, y, mx, my = self._load_split("val")
        B, L, _ = x.shape
        T = y.shape[1]
        if mx is None:
            mx = torch.zeros((B, L, 1), dtype=torch.float32)
        if my is None:
            my = torch.zeros((B, T, 1), dtype=torch.float32)
        self._val_ds = TensorDataset(x, mx, y, my)

        # ---- test ----
        x, y, mx, my = self._load_split("test")
        B, L, _ = x.shape
        T = y.shape[1]
        if mx is None:
            mx = torch.zeros((B, L, 1), dtype=torch.float32)
        if my is None:
            my = torch.zeros((B, T, 1), dtype=torch.float32)
        self._test_ds = TensorDataset(x, mx, y, my)

    def _collate(self, batch):
        var_x, marker_x, var_y, marker_y = zip(*batch)
        return (
            torch.stack(var_x, 0),
            torch.stack(marker_x, 0),
            torch.stack(var_y, 0),
            torch.stack(marker_y, 0),
        )

    def train_dataloader(self):
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self._test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate,
        )


# -------------------------
# Loss
# -------------------------
class PeakWeightedMSE(nn.Module):
    """
    Peakを重視するMSE（kWスケール想定）
    """
    def __init__(self, peak_threshold: float = 50.0, weight: float = 3.0):
        super().__init__()
        self.peak_threshold = float(peak_threshold)
        self.weight = float(weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = (pred - target) ** 2
        w = torch.ones_like(target)
        w[target > self.peak_threshold] = self.weight
        return (loss * w).mean()


# -------------------------
# Lightning Runner
# -------------------------
class Runner(LightningModule):
    def __init__(self, model: nn.Module, exp_conf: Dict):
        super().__init__()
        self.model = model
        self.conf = exp_conf

        self.pv_index = int(exp_conf.get("pv_index", 0))
        self.enforce_nonneg = bool(exp_conf.get("enforce_nonneg", False))

        self.loss_train = PeakWeightedMSE(
            peak_threshold=float(exp_conf.get("peak_threshold", 50.0)),
            weight=float(exp_conf.get("peak_weight", 3.0)),
        )

    def _predict(self, var_x, marker_x):
        outputs = self.model(var_x, marker_x)
        pred = _extract_pred_to_3d1(outputs, pv_index_if_multi=self.pv_index)
        if self.enforce_nonneg:
            pred = torch.clamp(pred, min=0.0)
        return pred

    def training_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = batch
        var_x = var_x.to(self.device).float()
        marker_x = marker_x.to(self.device).float()
        var_y = var_y.to(self.device).float()

        pred = self._predict(var_x, marker_x)
        loss = self.loss_train(pred, var_y)

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = batch
        var_x = var_x.to(self.device).float()
        marker_x = marker_x.to(self.device).float()
        var_y = var_y.to(self.device).float()

        pred = self._predict(var_x, marker_x)
        loss = F.mse_loss(pred, var_y)
        mae = F.l1_loss(pred, var_y)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self.log("val/mae", mae, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = batch
        var_x = var_x.to(self.device).float()
        marker_x = marker_x.to(self.device).float()
        var_y = var_y.to(self.device).float()

        pred = self._predict(var_x, marker_x)
        mse = F.mse_loss(pred, var_y)
        mae = F.l1_loss(pred, var_y)

        self.log("test/mse", mse, prog_bar=True, on_epoch=True)
        self.log("test/mae", mae, prog_bar=True, on_epoch=True)
        return mse

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.conf.get("lr", 1e-4)),
            weight_decay=float(self.conf.get("optimizer_weight_decay", 1e-4)),
        )
        sch = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=int(self.conf.get("lr_step_size", 20)),
            gamma=float(self.conf.get("lr_gamma", 0.5)),
        )
        return [opt], [sch]


# -------------------------
# Model factory
# -------------------------
def get_model(model_name: str, exp_conf: Dict) -> nn.Module:
    if model_name in ["iTransformerPeak", "iTransformer"]:
        from easytsf.model.iTransformer_peak import iTransformerPeak
        model_class = iTransformerPeak

    elif model_name == "MMK_Mix":
        from easytsf.model.MMK_Mix import MMK_Mix
        model_class = MMK_Mix

    elif model_name == "MMK_Mix_FusionPV":
        from easytsf.model.MMK_Mix_FusionPV import MMK_Mix_FusionPV
        model_class = MMK_Mix_FusionPV

    else:
        raise ValueError(f"[ERROR] Model {model_name} not supported.")

    sig = inspect.signature(model_class.__init__)
    valid = [p.name for p in sig.parameters.values() if p.name != "self"]
    kwargs = {k: v for k, v in exp_conf.items() if k in valid}
    return model_class(**kwargs)


# -------------------------
# Train
# -------------------------
def train(exp_conf: Dict, save_dir: str, devices: int, seed: int):
    seed_everything(seed, workers=True)

    # Data
    dm = NPYWindowDataModule(exp_conf)
    dm.setup()

    # Model
    model = get_model(exp_conf["model_name"], exp_conf)
    runner = Runner(model, exp_conf)

    # Logger
    logger = CSVLogger(save_dir=save_dir, name=str(exp_conf["model_name"]))

    # Callbacks
    callbacks = [TQDMProgressBar(refresh_rate=1)]

    if bool(exp_conf.get("enable_early_stop", True)):
        callbacks.append(
            EarlyStopping(
                monitor=str(exp_conf.get("val_metric", "val/loss")),
                patience=int(exp_conf.get("early_stop_patience", 15)),
                min_delta=float(exp_conf.get("early_stop_min_delta", 0.0)),
                mode="min",
            )
        )

    callbacks.append(
        ModelCheckpoint(
            monitor=str(exp_conf.get("val_metric", "val/loss")),
            mode="min",
            save_top_k=int(exp_conf.get("ckpt_save_top_k", 5)),
            save_last=bool(exp_conf.get("ckpt_save_last", True)),
            filename="best-epoch{epoch:02d}-val_loss{val/loss:.4f}",
        )
    )

    trainer = Trainer(
        max_epochs=int(exp_conf.get("max_epochs", 50)),
        accelerator="auto",
        devices=devices,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=float(exp_conf.get("gradient_clip_val", 1.0)),
        log_every_n_steps=int(exp_conf.get("log_every_n_steps", 50)),
    )

    trainer.fit(
        runner,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
    )

    print("\n==============================")
    print("  TEST EVALUATION (Best Model)")
    print("==============================")
    trainer.test(runner, dataloaders=dm.test_dataloader())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="config file path")
    parser.add_argument("-s", "--save_dir", default="save", help="save root dir")
    parser.add_argument("--devices", type=int, default=1, help="num devices")
    parser.add_argument("--seed", type=int, default=2025, help="seed")
    args = parser.parse_args()

    # import config module
    config_path = args.config
    if config_path.endswith(".py"):
        config_path = config_path[:-3]
    config_path = config_path.replace("/", ".").replace("\\", ".")
    config_module = importlib.import_module(config_path)

    exp_conf = config_module.exp_conf

    # 最低限の必須キー
    for k in ["data_dir", "model_name"]:
        if k not in exp_conf:
            raise KeyError(f"[ERROR] exp_conf missing key: {k}")

    train(exp_conf, save_dir=args.save_dir, devices=args.devices, seed=args.seed)


if __name__ == "__main__":
    main()
