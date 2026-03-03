# -*- coding: utf-8 -*-
"""
traingraph_24.py (FULL VERSION - FIXED)
- 24in/24out 向け学習スクリプト
- marker_y（未来GHI由来の昼夜）を使った mask 学習に対応

【今回の修正点（重要）】
- exp_conf に "loss_fn"（nn.Module）が入っている場合、それを最優先で使う
  → これで config に SmoothL1Loss を書けば必ず Huber になる
- Lightning の警告回避のため save_hyperparameters(ignore=[..., "loss_fn"]) を設定

使い方（例）
python traingraph_24.py -c config\\tsukuba_conf\\MMK_Mix_PV_24_fromseries_norm_huber.py -s save\\MMK_Mix_PV_24_fromseries_norm_huber_v1 --devices 1 --seed 2025
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
    """target が peak_threshold を超える点に weight を掛ける MSE"""
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
    """GHI をブーストする（必要な時だけ）。デフォルト 1.0（ブーストなし）"""
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
    """mask==1 の場所だけで MSE（pred/target/mask: (B,T,1)）"""
    diff2 = (pred - target) ** 2
    masked = diff2 * mask
    denom = mask.sum().clamp_min(1.0)
    return masked.sum() / denom


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask==1 の場所だけで MAE"""
    diff = (pred - target).abs()
    masked = diff * mask
    denom = mask.sum().clamp_min(1.0)
    return masked.sum() / denom


def masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, beta: float = 0.1) -> torch.Tensor:
    """mask==1 の場所だけで Huber (SmoothL1)"""
    diff = (pred - target).abs()
    b = float(beta)
    quad = 0.5 * (diff ** 2) / max(b, 1e-12)
    lin = diff - 0.5 * b
    hub = torch.where(diff < b, quad, lin)
    masked = hub * mask
    denom = mask.sum().clamp_min(1.0)
    return masked.sum() / denom


class Runner24(LightningModule):
    """
    24in/24out の学習Runner
    - marker_y（未来GHI由来）から day/night mask を作り、昼＋夜の重み付き損失で学習
    """
    def __init__(self, model, exp_conf: dict):
        super().__init__()
        # ★loss_fn が nn.Module の場合、Lightning が勝手に保存するので ignore して警告回避
        self.save_hyperparameters(ignore=["model", "loss_fn"])
        self.model = model
        self.config = exp_conf

        # ===== configから受け取る（無ければデフォルト）=====
        self.ghi_index = int(self.config.get("ghi_index", 2))
        self.ghi_boost = float(self.config.get("ghi_boost", 1.0))
        self.pv_index = int(self.config.get("pv_index", 0))

        # mask 学習
        self.use_day_mask = bool(self.config.get("use_day_mask", True))
        self.day_mask_threshold = float(self.config.get("day_mask_threshold", 0.5))

        # 損失の重み（夜も少し罰して “夜の定数出力” を防ぐ）
        self.day_loss_weight = float(self.config.get("day_loss_weight", 1.0))
        self.night_loss_weight = float(self.config.get("night_loss_weight", 0.2))

        # 制約（主に指標・可視化のブレを減らす）
        self.enforce_nonneg = bool(self.config.get("enforce_nonneg", True))
        self.force_night0 = bool(self.config.get("force_night0", True))
        self.apply_constraints_in_metrics = bool(self.config.get("apply_constraints_in_metrics", True))

        # loss 選択（loss_fn が指定されていれば最優先）
        self.loss_type = str(self.config.get("loss_type", "mse")).lower()
        self.huber_beta = float(self.config.get("huber_beta", 0.1))

        # ★修正：config の loss_fn を最優先で採用
        cfg_loss_fn = self.config.get("loss_fn", None)
        if isinstance(cfg_loss_fn, nn.Module):
            self.loss_fn = cfg_loss_fn
            # 表示/ログ用に loss_type を補助
            self.loss_type = self.loss_fn.__class__.__name__.lower()
        else:
            peak_th = float(self.config.get("peak_threshold", 30.0))
            peak_w = float(self.config.get("peak_weight", 3.0))

            if self.loss_type in ["peak", "peak_mse", "peakweightedmse"]:
                self.loss_fn = PeakWeightedMSE(peak_threshold=peak_th, weight=peak_w)
            elif self.loss_type in ["mae", "l1"]:
                self.loss_fn = nn.L1Loss()
            elif self.loss_type in ["huber", "smoothl1"]:
                self.loss_fn = nn.SmoothL1Loss(beta=self.huber_beta)
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
        """marker_y: (B,T,1) -> day_mask: (B,T,1)"""
        return (marker_y > self.day_mask_threshold).float()

    def _apply_constraints(self, pred_pv: torch.Tensor, day_mask: torch.Tensor) -> torch.Tensor:
        """pred>=0 と night=0 を適用（主に指標/可視化の安定化用）"""
        out = pred_pv
        if self.enforce_nonneg:
            out = torch.clamp(out, min=0.0)
        if self.force_night0:
            out = out * day_mask
        return out

    def _masked_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        mask==1 の箇所だけで loss を計算
        - loss_fn が SmoothL1Loss / L1Loss / MSELoss の場合はそれに対応
        """
        # loss_fn を見て分岐（確実に指定通り動かす）
        if isinstance(self.loss_fn, nn.SmoothL1Loss):
            beta = getattr(self.loss_fn, "beta", self.huber_beta)
            return masked_huber(pred, target, mask, beta=float(beta))
        if isinstance(self.loss_fn, nn.L1Loss):
            return masked_mae(pred, target, mask)
        if isinstance(self.loss_fn, PeakWeightedMSE):
            # PeakWeightedMSE の mask 版（簡易）
            diff2 = (pred - target) ** 2
            weights = torch.ones_like(target)
            peak_th = float(self.config.get("peak_threshold", 30.0))
            peak_w = float(self.config.get("peak_weight", 3.0))
            weights[target > peak_th] = peak_w
            masked = diff2 * weights * mask
            denom = mask.sum().clamp_min(1.0)
            return masked.sum() / denom

        # default: MSE
        return masked_mse(pred, target, mask)

    def training_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred_pv = self._predict_pv(var_x, marker_x)

        day_mask = self._make_day_mask(marker_y)
        night_mask = 1.0 - day_mask

        if self.use_day_mask:
            # ★学習でも制約を反映（平均解の逃げ道を減らす）
            pred_for_loss = self._apply_constraints(pred_pv, day_mask)

            loss_day = self._masked_loss(pred_for_loss, var_y, day_mask)
            loss_night = self._masked_loss(pred_for_loss, var_y, night_mask)
            loss = self.day_loss_weight * loss_day + self.night_loss_weight * loss_night
        else:
            pred_for_loss = pred_pv
            if self.enforce_nonneg:
                pred_for_loss = torch.clamp(pred_for_loss, min=0.0)
            # 全点 loss（loss_fn の種類に応じて）
            loss = self.loss_fn(pred_for_loss, var_y)

        if hasattr(self.model, "get_load_balancing_loss"):
            loss = loss + 0.1 * self.model.get_load_balancing_loss()

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred_pv = self._predict_pv(var_x, marker_x)

        day_mask = self._make_day_mask(marker_y)
        night_mask = 1.0 - day_mask

        pred_for_metrics = pred_pv
        if self.apply_constraints_in_metrics:
            pred_for_metrics = self._apply_constraints(pred_for_metrics, day_mask)
        else:
            if self.enforce_nonneg:
                pred_for_metrics = torch.clamp(pred_for_metrics, min=0.0)

        val_mse_all = F.mse_loss(pred_for_metrics, var_y)
        val_mae_all = F.l1_loss(pred_for_metrics, var_y)

        val_mse_day = masked_mse(pred_for_metrics, var_y, day_mask)
        val_mae_day = masked_mae(pred_for_metrics, var_y, day_mask)
        val_mse_night = masked_mse(pred_for_metrics, var_y, night_mask)
        val_mae_night = masked_mae(pred_for_metrics, var_y, night_mask)

        self.log("val/loss", val_mse_day, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mae", val_mae_day, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        self.log("val/mse_all", val_mse_all, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mae_all", val_mae_all, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mse_night", val_mse_night, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mae_night", val_mae_night, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True)

        return val_mse_day

    def test_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred_pv = self._predict_pv(var_x, marker_x)

        day_mask = self._make_day_mask(marker_y)
        night_mask = 1.0 - day_mask

        pred_for_metrics = pred_pv
        if self.apply_constraints_in_metrics:
            pred_for_metrics = self._apply_constraints(pred_for_metrics, day_mask)
        else:
            if self.enforce_nonneg:
                pred_for_metrics = torch.clamp(pred_for_metrics, min=0.0)

        mse_day = masked_mse(pred_for_metrics, var_y, day_mask)
        mae_day = masked_mae(pred_for_metrics, var_y, day_mask)
        mse_night = masked_mse(pred_for_metrics, var_y, night_mask)
        mae_night = masked_mae(pred_for_metrics, var_y, night_mask)

        mse_all = F.mse_loss(pred_for_metrics, var_y)
        mae_all = F.l1_loss(pred_for_metrics, var_y)

        self.log("test/mse_day", mse_day, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/mae_day", mae_day, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/mse_night", mse_night, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test/mae_night", mae_night, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test/mse_all", mse_all, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test/mae_all", mae_all, prog_bar=False, on_step=False, on_epoch=True)

        # 互換（既存ツールが test/mae を参照している場合）
        self.log("test/mse", mse_day, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test/mae", mae_day, prog_bar=False, on_step=False, on_epoch=True)

        return mse_day

    def configure_optimizers(self):
        lr = float(self.config.get("lr", 1e-4))
        weight_decay = float(self.config.get("optimizer_weight_decay", self.config.get("weight_decay", 1e-4)))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        step_size = int(self.config.get("lr_step_size", 15))
        gamma = float(self.config.get("lr_gamma", 0.5))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return [optimizer], [scheduler]


def get_model(model_name: str, exp_conf: dict):
    """config に書かれた引数だけをモデルに渡す（余計なキーで落ちないように）"""
    if model_name in ["iTransformerPeak", "iTransformer"]:
        from easytsf.model.iTransformer_peak import iTransformerPeak
        model_class = iTransformerPeak
    elif model_name == "MMK_Mix":
        from easytsf.model.MMK_Mix import MMK_Mix
        model_class = MMK_Mix
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
    if "val/mae_night" in df.columns:
        plt.plot(df["val/mae_night"].dropna(), label="Val MAE (Night)")
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
            EarlyStopping(monitor="val/loss", patience=patience, mode="min", verbose=True),
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
