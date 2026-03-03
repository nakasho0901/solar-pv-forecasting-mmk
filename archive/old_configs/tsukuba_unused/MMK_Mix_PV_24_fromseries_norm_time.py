# -*- coding: utf-8 -*-
"""
traingraph_24_time.py (REPLACEMENT)
- 24in/24out + MMK_Mix_Time 用の学習スクリプト
- marker_y を使った day/night 分離を「学習loss」にも反映
  => loss = w_day * loss(day) + w_night * loss(night)

狙い:
- 夜に変な定数を出す解（平均解）を学習で罰し、昼の形状学習へ誘導する
"""

import os
import sys
import argparse
import importlib.util
import inspect
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

# プロジェクトルートを import パスに追加
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from easytsf.runner.data_runner import NPYDataInterface
from easytsf.model.MMK_Mix_Time import MMK_Mix_Time


def load_config_module(config_path: str):
    spec = importlib.util.spec_from_file_location("exp_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"[ERROR] failed to load config: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_channel_boost(var_x: torch.Tensor, index: int, factor: float) -> torch.Tensor:
    """var_x[:, :, index] を factor 倍（必要な場合のみ）"""
    if factor == 1.0:
        return var_x
    x = var_x.clone()
    if x.dim() == 3 and 0 <= index < x.shape[-1]:
        x[:, :, index] = x[:, :, index] * factor
    return x


def _extract_pv(t: torch.Tensor, pv_index: int = 0) -> torch.Tensor:
    """
    (B,T,C) -> (B,T,1) にする。C=1 ならそのまま、C>1 なら pv_index を使う。
    """
    if t.dim() != 3:
        raise ValueError(f"[ERROR] tensor must be 3D (B,T,C). got {t.shape}")
    if t.shape[-1] == 1:
        return t
    idx = pv_index if (0 <= pv_index < t.shape[-1]) else 0
    return t[:, :, idx:idx + 1]


class LitMMKMixTime(pl.LightningModule):
    def __init__(self, exp_conf: Dict[str, Any]):
        super().__init__()
        # loss_fn は nn.Module なので hparams に入れない
        self.save_hyperparameters(ignore=["loss_fn"])

        self.exp_conf = exp_conf
        model_conf = exp_conf.get("model_conf", {})
        train_conf = exp_conf.get("train_conf", {})
        data_conf = exp_conf.get("data_conf", {})

        # ---- Model ----
        # configの不要キーがあっても落ちないように、MMK_Mix_Time.__init__ の引数だけ通す
        sig = inspect.signature(MMK_Mix_Time.__init__)
        valid = {p.name for p in sig.parameters.values() if p.name != "self"}
        filtered_model_conf = {k: v for k, v in model_conf.items() if k in valid}
        self.model = MMK_Mix_Time(**filtered_model_conf)

        # ---- Indices ----
        self.pv_index = int(data_conf.get("pv_index", 0))
        self.ghi_index = int(data_conf.get("ghi_index", 0))

        # ---- Train settings ----
        self.lr = float(train_conf.get("lr", 3e-4))
        self.weight_decay = float(train_conf.get("weight_decay", 0.0))

        self.loss_type = str(train_conf.get("loss_type", "huber")).lower()
        if self.loss_type in ["huber", "smoothl1"]:
            beta = float(train_conf.get("huber_beta", 1.0))
            self.loss_fn = nn.SmoothL1Loss(beta=beta, reduction="none")
        else:
            self.loss_fn = nn.MSELoss(reduction="none")

        # day/night weighting（これが今回のキモ）
        self.w_day = float(train_conf.get("w_day", 1.0))
        self.w_night = float(train_conf.get("w_night", 0.3))

        # marker_y threshold
        self.day_threshold = float(train_conf.get("day_threshold", 0.5))

        # optional post-process for metrics
        self.force_night0 = bool(train_conf.get("force_night0", False))
        self.enforce_nonneg = bool(train_conf.get("enforce_nonneg", False))

        # optional input boost
        self.ghi_boost = float(train_conf.get("ghi_boost", 1.0))

    def forward(self, x_enc, x_mark_enc):
        return self.model(x_enc, x_mark_enc)

    def _make_day_mask(self, marker_y: torch.Tensor) -> torch.Tensor:
        """
        marker_y: (B,T,1) 想定 -> bool mask (B,T,1)
        """
        if marker_y.dim() == 2:
            marker_y = marker_y.unsqueeze(-1)
        if marker_y.shape[-1] != 1:
            marker_y = marker_y[:, :, :1]
        return marker_y > self.day_threshold

    def _predict_raw(self, var_x: torch.Tensor, marker_x: torch.Tensor) -> torch.Tensor:
        x_in = _apply_channel_boost(var_x, self.ghi_index, self.ghi_boost)
        out = self.model(x_in, marker_x)
        pred = out[0] if isinstance(out, (tuple, list)) else out
        pred_pv = _extract_pv(pred, self.pv_index)
        return pred_pv

    def _postprocess_pred(self, pred_pv: torch.Tensor, day_mask: torch.Tensor) -> torch.Tensor:
        out = pred_pv
        if self.enforce_nonneg:
            out = torch.clamp(out, min=0.0)
        if self.force_night0:
            out = out.masked_fill(~day_mask, 0.0)
        return out

    def _weighted_daynight_loss(self, pred_pv: torch.Tensor, true_pv: torch.Tensor, day_mask: torch.Tensor):
        """
        loss = w_day * mean(loss on day) + w_night * mean(loss on night)
        """
        base = self.loss_fn(pred_pv, true_pv)  # (B,T,1)
        day = day_mask
        night = ~day_mask

        eps = 1e-8
        day_count = day.float().sum().clamp(min=eps)
        night_count = night.float().sum().clamp(min=eps)

        day_loss = (base * day.float()).sum() / day_count
        night_loss = (base * night.float()).sum() / night_count
        total = self.w_day * day_loss + self.w_night * night_loss
        return total, day_loss.detach(), night_loss.detach()

    def _mae_masked(self, pred: torch.Tensor, true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.float().sum().clamp(min=1.0)
        return (torch.abs(pred - true) * mask.float()).sum() / denom

    def training_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = batch
        true_pv = _extract_pv(var_y, self.pv_index)
        day_mask = self._make_day_mask(marker_y)

        pred_pv = self._predict_raw(var_x, marker_x)

        loss, loss_day, loss_night = self._weighted_daynight_loss(pred_pv, true_pv, day_mask)

        # metrics（学習中は raw pred でログ）
        with torch.no_grad():
            mae_all = torch.mean(torch.abs(pred_pv - true_pv))
            mae_day = self._mae_masked(pred_pv, true_pv, day_mask)
            mae_night = self._mae_masked(pred_pv, true_pv, ~day_mask)

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/loss_day", loss_day)
        self.log("train/loss_night", loss_night)
        self.log("train/mae_all", mae_all)
        self.log("train/mae_day", mae_day)
        self.log("train/mae_night", mae_night)
        return loss

    def validation_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = batch
        true_pv = _extract_pv(var_y, self.pv_index)
        day_mask = self._make_day_mask(marker_y)

        pred_pv = self._predict_raw(var_x, marker_x)

        loss, loss_day, loss_night = self._weighted_daynight_loss(pred_pv, true_pv, day_mask)

        pred_eval = self._postprocess_pred(pred_pv, day_mask)

        mae_all = torch.mean(torch.abs(pred_eval - true_pv))
        mae_day = self._mae_masked(pred_eval, true_pv, day_mask)
        mae_night = self._mae_masked(pred_eval, true_pv, ~day_mask)

        self.log("val/loss", loss, prog_bar=True)
        self.log("val/loss_day", loss_day)
        self.log("val/loss_night", loss_night)
        self.log("val/mae_all", mae_all)
        self.log("val/mae_day", mae_day)
        self.log("val/mae_night", mae_night)
        return loss

    def test_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = batch
        true_pv = _extract_pv(var_y, self.pv_index)
        day_mask = self._make_day_mask(marker_y)

        pred_pv = self._predict_raw(var_x, marker_x)

        loss, loss_day, loss_night = self._weighted_daynight_loss(pred_pv, true_pv, day_mask)

        pred_eval = self._postprocess_pred(pred_pv, day_mask)

        mae_all = torch.mean(torch.abs(pred_eval - true_pv))
        mae_day = self._mae_masked(pred_eval, true_pv, day_mask)
        mae_night = self._mae_masked(pred_eval, true_pv, ~day_mask)

        mse_all = torch.mean((pred_eval - true_pv) ** 2)
        mse_day = ((pred_eval - true_pv) ** 2 * day_mask.float()).sum() / (day_mask.float().sum().clamp(min=1.0))
        mse_night = ((pred_eval - true_pv) ** 2 * (~day_mask).float()).sum() / (((~day_mask).float().sum()).clamp(min=1.0))

        self.log("test/loss", loss)
        self.log("test/loss_day", loss_day)
        self.log("test/loss_night", loss_night)
        self.log("test/mae_all", mae_all)
        self.log("test/mae_day", mae_day)
        self.log("test/mae_night", mae_night)
        self.log("test/mse_all", mse_all)
        self.log("test/mse_day", mse_day)
        self.log("test/mse_night", mse_night)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-s", "--save", type=str, required=True)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    cfg = load_config_module(args.config)
    exp_conf = getattr(cfg, "exp_conf", None)
    if exp_conf is None:
        raise ValueError("[ERROR] exp_conf not found in config")

    # DataModule / DataInterface
    # NPYDataInterface が exp_conf 形式（または data_conf 形式）をどちらも受けられる想定で分岐
    data_conf = exp_conf.get("data_conf", None)
    if data_conf is not None:
        data = NPYDataInterface(data_conf)
    else:
        data = NPYDataInterface(**exp_conf)

    try:
        data.setup(stage="fit")
    except Exception:
        try:
            data.setup()
        except Exception:
            pass

    train_loader = data.train_dataloader()
    val_loader = data.val_dataloader()
    test_loader = data.test_dataloader()

    model = LitMMKMixTime(exp_conf)

    train_conf = exp_conf.get("train_conf", {})
    patience = int(train_conf.get("patience", 10))
    max_epochs = int(train_conf.get("max_epochs", 50))

    ckpt_cb = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        filename="best-epoch{epoch:02d}-val_loss{val/loss:.4f}",
    )
    es_cb = EarlyStopping(monitor="val/loss", mode="min", patience=patience)

    logger = CSVLogger(save_dir=args.save, name=exp_conf.get("model_name", "MMK_Mix_Time"))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=args.devices,
        callbacks=[ckpt_cb, es_cb],
        logger=logger,
        enable_checkpointing=True,
        log_every_n_steps=50,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    print("\n==============================")
    print("  TEST EVALUATION (Best Model)")
    print("==============================")
    trainer.test(model, dataloaders=test_loader, ckpt_path=ckpt_cb.best_model_path)

    print(f"[INFO] best ckpt: {ckpt_cb.best_model_path}")
    print(f"[INFO] log dir : {logger.log_dir}")


if __name__ == "__main__":
    main()
