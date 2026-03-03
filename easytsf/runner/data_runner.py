# easytsf/runner/data_runner.py
# -*- coding: utf-8 -*-
"""
データ読み込み用モジュール

- 既存の CSV / npz 版 DataInterface（GeneralTSF 用）はそのまま残しておく
- PVProcessed(96→96) 用には、窓切り済み .npy を読む NPYDataInterface を追加

【hardzero 対応】
- dataset_PV/processed_hardzero にある
  train/val/test_marker_y.npy（ghi_future）を読み込めるようにする
- Lightning Runner 側（exp_runner.forward）が (var_x, marker_x, var_y, marker_y) を受け取るため、
  第4要素 = marker_y に hardzero の marker_y を渡す
- marker_y が無い場合は従来通り y_mark（なければゼロ埋め）を渡す
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as L

# =====================================
#  既存: GeneralTSF 用 DataInterface
# =====================================
class GeneralTSFDataset(Dataset):
    """元々の easytsf 用汎用 Dataset（そのまま残しておく）"""

    def __init__(self, hist_len, pred_len, variable, time_feature):
        super().__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.variable = variable
        self.time_feature = time_feature

    def __len__(self):
        # hist_len を考慮したスライディング窓数
        return self.variable.shape[0] - self.hist_len - self.pred_len + 1

    def __getitem__(self, idx):
        # x: [hist_len, C], y: [pred_len, C]
        s = idx
        e = idx + self.hist_len
        x = self.variable[s:e]                  # [H, Cx]
        x_mark = self.time_feature[s:e]         # [H, Ctx]

        s_y = e
        e_y = e + self.pred_len
        y = self.variable[s_y:e_y]              # [F, Cx]
        y_mark = self.time_feature[s_y:e_y]     # [F, Ctx]
        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(x_mark).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(y_mark).float(),
        )


class DataInterface(L.LightningDataModule):
    """
    既存 CSV / npz データ用 DataModule（今回は使っていないが互換のため残す）
    """

    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.hist_len = int(self.hparams.hist_len)
        self.pred_len = int(self.hparams.pred_len)
        self.batch_size = int(getattr(self.hparams, "batch_size", 64))
        self.num_workers = int(getattr(self.hparams, "num_workers", 0))

        # ここでは self.variable / self.time_feature がすでに numpy で用意されている前提
        self.variable = self.hparams.variable
        self.time_feature = self.hparams.time_feature
        self.train_len = int(self.hparams.train_len)
        self.val_len = int(self.hparams.val_len)

    def train_dataloader(self):
        return DataLoader(
            dataset=GeneralTSFDataset(
                self.hist_len,
                self.pred_len,
                self.variable[:self.train_len].copy(),
                self.time_feature[:self.train_len].copy(),
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=GeneralTSFDataset(
                self.hist_len,
                self.pred_len,
                self.variable[self.train_len - self.hist_len:self.train_len + self.val_len].copy(),
                self.time_feature[self.train_len - self.hist_len:self.train_len + self.val_len].copy(),
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=GeneralTSFDataset(
                self.hist_len,
                self.pred_len,
                self.variable[self.train_len + self.val_len - self.hist_len:].copy(),
                self.time_feature[self.train_len + self.val_len - self.hist_len:].copy(),
            ),
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
        )


# =====================================
#  追加: PVProcessed 用 NPYDataInterface
# =====================================

class _NPYWindowDataset(Dataset):
    """
    窓切り済み .npy をそのまま使う Dataset

    想定ファイル:
        {root}/train_x.npy : [N, H, Cx]
        {root}/train_y.npy : [N, F] or [N, F, 1]
        {root}/train_x_mark.npy (任意): [N, H, 4]
        {root}/train_y_mark.npy (任意): [N, F, 4]
        {root}/train_marker_y.npy (任意/hardzero): [N, F, 1]  ← GHI future
    """

    def __init__(self, root: str, split: str):
        super().__init__()
        x_path = os.path.join(root, f"{split}_x.npy")
        y_path = os.path.join(root, f"{split}_y.npy")
        if not os.path.exists(x_path):
            raise FileNotFoundError(f"入力特徴量ファイルが見つかりません: {x_path}")
        if not os.path.exists(y_path):
            raise FileNotFoundError(f"ターゲットファイルが見つかりません: {y_path}")

        x = np.load(x_path)  # [N, H, Cx]
        y = np.load(y_path)  # [N, F] or [N, F, 1]

        # y の次元を [N, F, 1] に統一
        if y.ndim == 2:
            y = y[..., None]

        # 時刻特徴（存在しない場合はゼロ埋め）
        x_mark_path = os.path.join(root, f"{split}_x_mark.npy")
        y_mark_path = os.path.join(root, f"{split}_y_mark.npy")

        if os.path.exists(x_mark_path):
            x_mark = np.load(x_mark_path)
        else:
            # [N, H, 4] を想定（4つの時間特徴）
            x_mark = np.zeros((x.shape[0], x.shape[1], 4), dtype=np.float32)

        if os.path.exists(y_mark_path):
            y_mark = np.load(y_mark_path)
        else:
            y_mark = np.zeros((y.shape[0], y.shape[1], 4), dtype=np.float32)

        # hardzero: marker_y（ghi_future）
        marker_y_path = os.path.join(root, f"{split}_marker_y.npy")
        if os.path.exists(marker_y_path):
            marker_y = np.load(marker_y_path)
            if marker_y.ndim == 2:
                marker_y = marker_y[..., None]
        else:
            marker_y = None

        # Tensor 変換
        self.x = torch.from_numpy(x).float()
        self.x_mark = torch.from_numpy(x_mark).float()
        self.y = torch.from_numpy(y).float()
        self.y_mark = torch.from_numpy(y_mark).float()
        self.marker_y = torch.from_numpy(marker_y).float() if marker_y is not None else None

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        # exp_runner.forward は (var_x, marker_x, var_y, marker_y) の4要素を想定
        # marker_y があればそれを渡し、無ければ従来通り y_mark を渡す
        if self.marker_y is None:
            return self.x[idx], self.x_mark[idx], self.y[idx], self.y_mark[idx]
        else:
            return self.x[idx], self.x_mark[idx], self.y[idx], self.marker_y[idx]


class NPYDataInterface(L.LightningDataModule):
    """
    窓切り済み .npy をそのまま DataLoader に流す DataModule
    """

    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()
        self.data_dir = self.hparams.data_dir
        self.batch_size = int(getattr(self.hparams, "batch_size", 64))
        self.num_workers = int(getattr(self.hparams, "num_workers", 0))

    def setup(self, stage=None):
        self.train_set = _NPYWindowDataset(self.data_dir, "train")
        self.val_set = _NPYWindowDataset(self.data_dir, "val")
        self.test_set = _NPYWindowDataset(self.data_dir, "test")

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
