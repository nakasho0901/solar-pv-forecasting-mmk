# -*- coding: utf-8 -*-
"""
traingraph_24_time.py
- traingraph_24.py を壊さずに、MMK_Mix_Time を追加対応した版（完全新規ファイル）
- 24in/24out 向け学習スクリプト
- v2 marker_y（未来GHI由来の昼夜）を学習に反映するため、デフォルトで「昼だけ」でloss計算

使い方（例）
python traingraph_24_time.py -c config\\tsukuba_conf\\MMK_Mix_PV_24_fromseries_norm_time.py -s save\\MMK_Mix_Time_PV_24_fromseries_norm_huber_time_v1 --devices 1 --seed 2025

config 側で調整できる項目（任意）
- use_day_mask: bool (default True)  -> True: 昼だけでloss計算 / False: 全点でloss計算
- day_mask_threshold: float (default 0.5)
- loss_type: "mse" or "peak_mse" (default "mse")
- peak_threshold: float (default 30.0)
- peak_weight: float (default 3.0)
- ghi_boost: float (default 1.0)  -> 1.0推奨（まず安定）
- pv_index: int (default 0)        -> 出力が多変数のときPVのindex
"""

import os
import sys
import argparse
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import matplotlib.pyplot as plt

from lightning.pytorch import Trainer, seed_everything, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

# ===== パス設定 =====
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from easytsf.runner.data_runner import NPYDataInterface


# ===== ピーク重み付き損失（任意）=====
class PeakWeightedMSE(nn.Module):
    """
    target が peak_threshold を超える点に weight を掛ける MSE
    """
    def __init__(self, peak_threshold: float = 30.0, weight: float = 3.0):
        super().__init__()
        self.peak_threshold = float(peak_threshold)
        self.weight = float(weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = (pred - target) ** 2
        weights = torch.ones_like(target)
        weights[target > self.peak_threshold] = self.weight
        return (loss * weights).mean()


def _apply_ghi_boost(var_x: torch.Tensor, ghi_index: int = 2, factor: float = 1.0) -> torch.Tensor:
    """
    GHI をブーストする（必要な時だけ）。
    ※デフォルトは 1.0（=ブーストなし）推奨
    """
    if factor == 1.0:
        return var_x
    x = var_x.clone()
    if x.dim() == 3 and x.shape[-1] > ghi_index:
        x[:, :, ghi_index] = x[:, :, ghi_index] * factor
    return x


def _extract_pv_pred(pred: torch.Tensor, pv_index_if_multi: int = 0) -> torch.Tensor:
    """
    予測 pred から PV 1本 (B,T,1) を取り出す。

    - pred が (B,T,1): そのまま
    - pred が (B,T,C>=2): pv_index_if_multi を優先
    """
    if pred.dim() != 3:
        raise ValueError(f"[ERROR] pred shape unexpected: {tuple(pred.shape)} (expected 3D)")

    C = pred.shape[-1]
    if C == 1:
        return pred

    idx = int(pv_index_if_multi)
    if idx < 0 or idx >= C:
        idx = 0
    return pred[:, :, idx:idx + 1]


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    mask==1 の場所だけで MSE を計算（安定版）
    pred/target/mask: (B,T,1)
    """
    diff2 = (pred - target) ** 2
    masked = diff2 * mask
    denom = mask.sum().clamp_min(1.0)  # 0除算防止
    return masked.sum() / denom


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    mask==1 の場所だけで MAE を計算（安定版）
    """
    diff = (pred - target).abs()
    masked = diff * mask
    denom = mask.sum().clamp_min(1.0)
    return masked.sum() / denom


class Runner24(LightningModule):
    """
    24in/24out 向けの学習Runner
    - v2 marker_y（未来GHI由来）を使って「昼だけloss」(default) が可能
    """
    def __init__(self, model, exp_conf: dict):
        super().__init__()
        # loss_fn はModuleなので ignore して warning を消す
        self.save_hyperparameters(ignore=["model", "loss_fn"])
        self.model = model
        self.config = exp_conf

        # ===== configから受け取る（無ければデフォルト）=====
        self.ghi_index = int(self.config.get("ghi_index", 2))
        self.ghi_boost = float(self.config.get("ghi_boost", 1.0))
        self.pv_index = int(self.config.get("pv_index", 0))

        # 昼マスクを loss に使う
        self.use_day_mask = bool(self.config.get("use_day topics_mask", True)) if "use_day topics_mask" in self.config else bool(self.config.get("use_day_mask", True))
        self.day_mask_threshold = float(self.config.get("day_mask_threshold", 0.5))

        # loss 選択（デフォルトは mse）
        self.loss_type = str(self.config.get("loss_type", "mse")).lower()
        peak_th = float(self.config.get("peak_threshold", 30.0))
        peak_w  = float(self.config.get("peak_weight", 3.0))

        # ★ config で loss_fn（例：SmoothL1Loss）が渡されていれば最優先
        if "loss_fn" in self.config and isinstance(self.config["loss_fn"], nn.Module):
            self.loss_fn = self.config["loss_fn"]
        else:
            if self.loss_type in ["peak", "peak_mse", "peakweightedmse"]:
                self.loss_fn = PeakWeightedMSE(peak_threshold=peak_th, weight=peak_w)
            else:
                self.loss_fn = nn.MSELoss()

    def forward(self, x_enc, x_mark_enc):
        return self.model(x_enc, x_mark_enc)

    def _predict_pv(self, var_x: torch.Tensor, marker_x: torch.Tensor) -> torch.Tensor:
        x_in = _apply_ghi_boost(var_x, ghi_index=self.ghi_index, factor=self.ghi_boost)
        outputs = self.model(x_in, marker_x)
        pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        pred_pv = _extract_pv_pred(pred, pv_index_if_multi=self.pv_index)
        return pred_pv

    def _make_day_mask(self, marker_y: torch.Tensor) -> torch.Tensor:
        """
        marker_y: (B,T,1)
        day_mask: (B,T,1) float32 (day=1, night=0)
        """
        return (marker_y > self.day_mask_threshold).float()

    def training_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred_pv = self._predict_pv(var_x, marker_x)

        if self.use_day_mask:
            day_mask = self._make_day_mask(marker_y)
            loss = masked_mse(pred_pv, var_y, day_mask)
        else:
            loss = self.loss_fn(pred_pv, var_y)

        if hasattr(self.model, "get_load_balancing_loss"):
            loss = loss + 0.1 * self.model.get_load_balancing_loss()

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred_pv = self._predict_pv(var_x, marker_x)

        val_mse_all = F.mse_loss(pred_pv, var_y)
        val_mae_all = F.l1_loss(pred_pv, var_y)

        day_mask = self._make_day_mask(marker_y)
        val_mse_day = masked_mse(pred_pv, var_y, day_mask)
        val_mae_day = masked_mae(pred_pv, var_y, day_mask)

        self.log("val/loss", val_mse_day, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mae",  val_mae_day, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mse_all", val_mse_all, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mae_all", val_mae_all, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return val_mse_day

    def test_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred_pv = self._predict_pv(var_x, marker_x)

        day_mask = self._make_day_mask(marker_y)
        mse_day = masked_mse(pred_pv, var_y, day_mask)
        mae_day = masked_mae(pred_pv, var_y, day_mask)

        mse_all = F.mse_loss(pred_pv, var_y)
        mae_all = F.l1_loss(pred_pv, var_y)

        self.log("test/mse_day", mse_day, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/mae_day", mae_day, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/mse_all", mse_all, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test/mae_all", mae_all, prog_bar=False, on_step=False, on_epoch=True)

        self.log("test/mse", mse_day, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test/mae", mae_day, prog_bar=False, on_step=False, on_epoch=True)

        return mse_day

    def configure_optimizers(self):
        lr = float(self.config.get("lr", 1e-4))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        step_size = int(self.config.get("lr_step_size", 15))
        gamma = float(self.config.get("lr_gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return [optimizer], [scheduler]


def get_model(model_name: str, exp_conf: dict):
    """
    config に書かれた引数だけをモデルに渡す（余計なキーで落ちないように）
    """
    if model_name in ["iTransformerPeak", "iTransformer"]:
        from easytsf.model.iTransformer_peak import iTransformerPeak
        model_class = iTransformerPeak

    elif model_name == "MMK_Mix":
        from easytsf.model.MMK_Mix import MMK_Mix
        model_class = MMK_Mix

    elif model_name == "MMK_Mix_Time":
        # ★新規追加：marker_x を実際に使う版
        from easytsf.model.MMK_Mix_Time import MMK_Mix_Time
        model_class = MMK_Mix_Time

    else:
        raise ValueError(f"[ERROR] Model {model_name} not supported.")

    sig = inspect.signature(model_class.__init__)
    valid_params = [p.name for p in sig.parameters.values() if p.name != "self"]
    filtered_conf = {k: v for k, v in exp_conf.items() if k in valid_params}
    return model_class(**filtered_conf)


def save_plots(log_dir: str):
    metrics_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        return

    df = pd.read_csv(metrics_path)

    plt.figure()
    if "train/loss_epoch" in df.columns:
        plt.plot(df["train/loss_epoch"].dropna(), label="Train Loss")
    if "val/loss" in df.columns:
        plt.plot(df["val/loss"].dropna(), label="Val Loss (Day)")
    plt.legend()
    plt.title("Loss Epoch")
    plt.savefig(os.path.join(log_dir, "loss_epoch.png"))
    print("[INFO] loss-epoch グラフを保存しました。")

    plt.figure()
    if "val/mae" in df.columns:
        plt.plot(df["val/mae"].dropna(), label="Val MAE (Day)")
    if "val/mae_all" in df.columns:
        plt.plot(df["val/mae_all"].dropna(), label="Val MAE (All)")
    plt.legend()
    plt.title("MAE Epoch")
    plt.savefig(os.path.join(log_dir, "mae_epoch.png"))
    print("[INFO] MAE-epoch グラフを保存しました。")


def train_func(training_conf: dict, exp_conf: dict):
    seed_everything(training_conf["seed"])

    base_model = get_model(exp_conf["model_name"], exp_conf)
    runner = Runner24(base_model, exp_conf)
    data_module = NPYDataInterface(**exp_conf)

    logger = CSVLogger(save_dir=training_conf["save_dir"], name=exp_conf["model_name"])

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="best-epoch{epoch:02d}-val_loss{val/loss:.4f}",
        save_top_k=5,
        mode="min",
        save_last=True,
    )

    patience = int(exp_conf.get("early_stop_patience", 20))
    max_epochs = int(exp_conf.get("max_epochs", 60))

    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=training_conf["devices"],
        logger=logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            checkpoint_callback,
            EarlyStopping(
                monitor="val/loss",
                patience=patience,
                mode="min",
                verbose=True
            ),
        ],
        gradient_clip_val=float(exp_conf.get("gradient_clip_val", 1.0)),
    )

    trainer.fit(model=runner, datamodule=data_module)

    print("\n" + "=" * 30)
    print("  TEST EVALUATION (Best Model)")
    print("=" * 30)
    trainer.test(model=runner, datamodule=data_module, ckpt_path="best")

    save_plots(logger.log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-s", "--save_dir", type=str, default="lightning_logs")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("config", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    train_func(
        {"save_dir": args.save_dir, "devices": args.devices, "seed": args.seed},
        config_module.exp_conf
    )
