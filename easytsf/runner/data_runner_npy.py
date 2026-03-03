# easytsf/runner/data_runner_npy.py
# -*- coding: utf-8 -*-
"""
窓切り済みの .npy（train_x.npy 等）をそのまま DataLoader に供給する DataModule。

exp_runner.forward(var_x, marker_x, var_y, marker_y) に合わせて
( x, marker_x, y, marker_y ) の4タプルを返す。

【hardzero 対応】
- dataset_dir に train/val/test_marker_y.npy があれば marker_y として返す
- 無ければ従来通りダミー（0埋め）
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as L


class _NpyWindowDataset(Dataset):
    """X:[N,T,F], Y:[N,T] or [N,T,1], marker_y(optional):[N,T,1] を返すDataset。"""

    def __init__(self, x_path: str, y_path: str, marker_y_path: str = None):
        self.X = np.load(x_path)  # float32想定 [N,T,F]
        self.Y = np.load(y_path)  # float32想定 [N,T] or [N,T,1]
        assert self.X.shape[0] == self.Y.shape[0], "XとYのサンプル数が一致しません。"
        self.T = self.X.shape[1]  # 例: 96
        self.F = self.X.shape[2]  # 例: 8

        # y を [N,T,1] に統一
        if self.Y.ndim == 2:
            self.Y = self.Y[..., None]

        # marker_y（hardzero）
        self.marker_y = None
        if marker_y_path and os.path.exists(marker_y_path):
            my = np.load(marker_y_path)
            if my.ndim == 2:
                my = my[..., None]
            self.marker_y = my

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # x: [T,F], y: [T,1]
        x = torch.from_numpy(self.X[idx])               # [T,F]
        y = torch.from_numpy(self.Y[idx]).float()       # [T,1]

        # ダミーの marker_x（iTransformer 側で使わなくても形は必要）
        marker_x = torch.zeros((self.T, self.F), dtype=x.dtype)

        # marker_y（hardzero があればそれ / 無ければゼロ埋め）
        if self.marker_y is None:
            marker_y = torch.zeros((self.T, 1), dtype=y.dtype)
        else:
            marker_y = torch.from_numpy(self.marker_y[idx]).float()  # [T,1]

        return x.float(), marker_x.float(), y, marker_y


class NPYDataInterface(L.LightningDataModule):
    """
    exp_conf に:
      data_format="npy"
      data_dir="dataset_PV/processed" or "dataset_PV/processed_hardzero"
      train_x="train_x.npy", ... など
    が渡されている前提で使う。
    """
    def __init__(self, **conf):
        super().__init__()
        self.conf = conf
        d = conf.get("data_dir", "dataset_PV/processed")

        self.paths = {
            "train_x": os.path.join(d, conf.get("train_x", "train_x.npy")),
            "train_y": os.path.join(d, conf.get("train_y", "train_y.npy")),
            "val_x":   os.path.join(d, conf.get("val_x",   "val_x.npy")),
            "val_y":   os.path.join(d, conf.get("val_y",   "val_y.npy")),
            "test_x":  os.path.join(d, conf.get("test_x",  "test_x.npy")),
            "test_y":  os.path.join(d, conf.get("test_y",  "test_y.npy")),
            # hardzero marker（存在しなければ Dataset 側で無視）
            "train_marker_y": os.path.join(d, conf.get("train_marker_y", "train_marker_y.npy")),
            "val_marker_y":   os.path.join(d, conf.get("val_marker_y",   "val_marker_y.npy")),
            "test_marker_y":  os.path.join(d, conf.get("test_marker_y",  "test_marker_y.npy")),
        }

        self.batch_size   = int(conf.get("batch_size", 64))
        self.num_workers  = int(conf.get("num_workers", 0))
        self.persistent   = bool(conf.get("persistent_workers", False))
        self.pin_memory   = bool(conf.get("pin_memory", False))
        self.drop_last_train = bool(conf.get("drop_last_train", True))

    def setup(self, stage=None):
        self.train_ds = _NpyWindowDataset(
            self.paths["train_x"],
            self.paths["train_y"],
            self.paths.get("train_marker_y"),
        )
        self.val_ds = _NpyWindowDataset(
            self.paths["val_x"],
            self.paths["val_y"],
            self.paths.get("val_marker_y"),
        )
        self.test_ds = _NpyWindowDataset(
            self.paths["test_x"],
            self.paths["test_y"],
            self.paths.get("test_marker_y"),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=self.drop_last_train,
            num_workers=self.num_workers,
            persistent_workers=self.persistent and self.num_workers > 0,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent and self.num_workers > 0,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent and self.num_workers > 0,
            pin_memory=self.pin_memory,
        )
