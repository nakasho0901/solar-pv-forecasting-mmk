# -*- coding: utf-8 -*-
r"""
traingraph_24_24_pvin.py
------------------------------------------------------------
24-24 / PV-input (pvin) / MMK_Mix 用の学習スクリプト（完全版）

重要：
- NPYDataInterface から返る marker_y が (B,T,4) 等の「時刻マーカー」になる環境がある
- その場合、marker_y を daymask として使うと MAE=0 などの異常が出る
- そこで daymask は「入力Xに含めた is_daylight 特徴」から生成する（推奨）

設定：
- config["use_input_daymask"]=True のとき
    mask = var_x[:,:,daylight_feature_index] を (B,T) で使う
- False のときは marker_y を使う（0/1チャネル探索→無ければ全1）
------------------------------------------------------------
"""

import os
import sys
import argparse
import inspect
import torch
import torch.nn as nn

from lightning.pytorch import Trainer, seed_everything, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from easytsf.runner.data_runner import NPYDataInterface


# =========================
# Shape utilities
# =========================
def _to_BTL_1(x: torch.Tensor, B: int, T: int, name: str) -> torch.Tensor:
    """x を (B, T, 1) に揃える"""
    if x.dim() == 2:
        if x.shape == (B, T):
            return x.unsqueeze(-1)
        if x.shape == (T, B):
            return x.t().unsqueeze(-1)
        raise RuntimeError(f"[{name}] unexpected 2D shape: {tuple(x.shape)} (B={B},T={T})")

    if x.dim() == 3:
        if x.shape == (B, T, 1):
            return x
        if x.shape == (T, B, 1):
            return x.permute(1, 0, 2).contiguous()
        if x.shape == (B, 1, T):
            return x.permute(0, 2, 1).contiguous()
        raise RuntimeError(f"[{name}] unexpected 3D shape: {tuple(x.shape)} (B={B},T={T})")

    raise RuntimeError(f"[{name}] unexpected dim: {x.dim()} shape={tuple(x.shape)}")


def _pred_to_BTN(pred: torch.Tensor, B: int, T: int, name: str) -> torch.Tensor:
    """pred を (B, T, N) に揃える（(T,B,N) も許容）"""
    if pred.dim() != 3:
        raise RuntimeError(f"[{name}] pred must be 3D. got {pred.dim()} shape={tuple(pred.shape)}")

    if pred.shape[0] == B and pred.shape[1] == T:
        return pred
    if pred.shape[0] == T and pred.shape[1] == B:
        return pred.permute(1, 0, 2).contiguous()

    raise RuntimeError(f"[{name}] unexpected pred shape: {tuple(pred.shape)} (B={B},T={T})")


def _is_binary_like(v: torch.Tensor) -> bool:
    """0/1っぽいチャネル判定（厳しめ）"""
    v = v.detach()
    v_min = float(v.min())
    v_max = float(v.max())
    if v_min < -0.05 or v_max > 1.05:
        return False
    near0 = (v < 0.1).float().mean()
    near1 = (v > 0.9).float().mean()
    return float(near0 + near1) > 0.95


def make_mask_from_marker_or_ones(marker_y: torch.Tensor, B: int, T: int, prefer_index=None) -> torch.Tensor:
    """
    marker_y から daymask(0/1) を作る。
    - (B,T) or (B,T,1) : そのまま
    - (B,T,K) : binary-likeチャネルを探索、無ければ全1
    """
    if prefer_index is not None:
        prefer_index = int(prefer_index)

    # (B,T)
    if marker_y.dim() == 2:
        if marker_y.shape == (B, T):
            return (marker_y > 0.5).float()
        if marker_y.shape == (T, B):
            return (marker_y.t() > 0.5).float()
        return torch.ones((B, T), device=marker_y.device, dtype=marker_y.dtype)

    # (B,T,1) or (B,T,K)
    if marker_y.dim() == 3:
        # (T,B,K) -> (B,T,K)
        if marker_y.shape[0] == T and marker_y.shape[1] == B:
            my = marker_y.permute(1, 0, 2).contiguous()
        elif marker_y.shape[0] == B and marker_y.shape[1] == T:
            my = marker_y
        else:
            return torch.ones((B, T), device=marker_y.device, dtype=marker_y.dtype)

        K = my.shape[-1]
        if K == 1:
            return (my[:, :, 0] > 0.5).float()

        if prefer_index is not None and 0 <= prefer_index < K:
            return (my[:, :, prefer_index] > 0.5).float()

        for k in range(K):
            cand = my[:, :, k]
            if _is_binary_like(cand):
                return (cand > 0.5).float()

        return torch.ones((B, T), device=marker_y.device, dtype=marker_y.dtype)

    return torch.ones((B, T), device=marker_y.device, dtype=marker_y.dtype)


# =========================
# Loss
# =========================
class DayMaskedMSE(nn.Module):
    """pred,target: (B,T,1), mask: (B,T)"""
    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
        se = (pred - target) ** 2  # (B,T,1)
        weighted = se.squeeze(-1) * mask_bt  # (B,T)
        denom = torch.clamp(mask_bt.sum(), min=1.0)
        return weighted.sum() / denom


# =========================
# Runner
# =========================
class Runner24PV(LightningModule):
    def __init__(self, model, config: dict):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.config = config

        self.pv_index = int(config["pv_index"])      # 0
        self.ghi_index = int(config["ghi_index"])    # 3
        self.ghi_boost = float(config.get("ghi_boost", 1.0))

        # ★ ここが今回の本命
        self.use_input_daymask = bool(config.get("use_input_daymask", True))
        self.daylight_feature_index = int(config.get("daylight_feature_index", 8))  # is_daylight位置（確定=8）
        self.marker_y_day_index = config.get("marker_y_day_index", None)

        self.loss_fn = DayMaskedMSE()
        self.abs_err = nn.L1Loss(reduction="none")

    def _apply_ghi_boost(self, x: torch.Tensor) -> torch.Tensor:
        if self.ghi_boost == 1.0:
            return x
        if self.ghi_index < 0 or self.ghi_index >= x.shape[-1]:
            raise RuntimeError(f"[ghi_index] out of range: ghi_index={self.ghi_index}, F={x.shape[-1]}")
        x2 = x.clone()
        x2[:, :, self.ghi_index] *= self.ghi_boost
        return x2

    def _predict_pv(self, var_x: torch.Tensor, marker_x: torch.Tensor) -> torch.Tensor:
        B = var_x.shape[0]
        T = int(self.config["pred_len"])

        x_in = self._apply_ghi_boost(var_x)
        out = self.model(x_in, marker_x)
        pred = out[0] if isinstance(out, (tuple, list)) else out

        pred_btn = _pred_to_BTN(pred, B, T, "pred")

        N = pred_btn.shape[-1]
        if self.pv_index < 0 or self.pv_index >= N:
            raise RuntimeError(f"[pv_index] out of range: pv_index={self.pv_index}, N={N}")

        return pred_btn[:, :, self.pv_index:self.pv_index + 1]  # (B,T,1)

    def _make_daymask(self, var_x: torch.Tensor, marker_y: torch.Tensor) -> torch.Tensor:
        """(B,T)のmaskを返す"""
        B = var_x.shape[0]
        T = int(self.config["pred_len"])

        if self.use_input_daymask:
            # var_x: (B,T,F)
            if self.daylight_feature_index < 0 or self.daylight_feature_index >= var_x.shape[-1]:
                raise RuntimeError(
                    f"[daylight_feature_index] out of range: idx={self.daylight_feature_index}, F={var_x.shape[-1]}"
                )
            m = var_x[:, :, self.daylight_feature_index]
            return (m > 0.5).float()

        # marker_yから作る（binary-like探索、無ければ全1）
        return make_mask_from_marker_or_ones(marker_y, B, T, prefer_index=self.marker_y_day_index)

    def training_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [b.float() for b in batch]
        B = var_x.shape[0]
        T = int(self.config["pred_len"])

        pred = self._predict_pv(var_x, marker_x)
        target = _to_BTL_1(var_y, B, T, "var_y")
        mask_bt = self._make_daymask(var_x, marker_y)

        loss = self.loss_fn(pred, target, mask_bt)

        if hasattr(self.model, "get_load_balancing_loss"):
            loss = loss + 0.1 * self.model.get_load_balancing_loss()

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [b.float() for b in batch]
        B = var_x.shape[0]
        T = int(self.config["pred_len"])

        pred = self._predict_pv(var_x, marker_x)
        target = _to_BTL_1(var_y, B, T, "var_y")
        mask_bt = self._make_daymask(var_x, marker_y)

        loss = self.loss_fn(pred, target, mask_bt)

        mae_bt = self.abs_err(pred.squeeze(-1), target.squeeze(-1))  # (B,T)
        mae = (mae_bt * mask_bt).sum() / torch.clamp(mask_bt.sum(), min=1.0)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mae", mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [b.float() for b in batch]
        B = var_x.shape[0]
        T = int(self.config["pred_len"])

        pred = self._predict_pv(var_x, marker_x)
        target = _to_BTL_1(var_y, B, T, "var_y")
        mask_bt = self._make_daymask(var_x, marker_y)

        mse = self.loss_fn(pred, target, mask_bt)

        mae_bt = self.abs_err(pred.squeeze(-1), target.squeeze(-1))
        mae = (mae_bt * mask_bt).sum() / torch.clamp(mask_bt.sum(), min=1.0)

        self.log("test/mse", mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test/mae", mae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return mse

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.config.get("lr", 1e-4)),
            weight_decay=float(self.config.get("optimizer_weight_decay", 0.0)),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(self.config.get("lr_step_size", 15)),
            gamma=float(self.config.get("lr_gamma", 0.5)),
        )
        return [optimizer], [scheduler]


# =========================
# Model loader
# =========================
def get_model(model_name: str, exp_conf: dict):
    if model_name in ["MMK_Mix_24", "MMK_Mix"]:
        from easytsf.model.MMK_Mix_24 import MMK_Mix_24
        model_class = MMK_Mix_24
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    sig = inspect.signature(model_class.__init__)
    valid_params = [p.name for p in sig.parameters.values() if p.name != "self"]
    filtered = {k: v for k, v in exp_conf.items() if k in valid_params}
    return model_class(**filtered)


def train_func(training_conf: dict, exp_conf: dict):
    seed_everything(training_conf["seed"])

    model = get_model(exp_conf["model_name"], exp_conf)
    runner = Runner24PV(model, exp_conf)
    data_module = NPYDataInterface(**exp_conf)

    logger = CSVLogger(save_dir=training_conf["save_dir"], name=str(exp_conf["model_name"]))

    ckpt = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="best-epoch{epoch:02d}-val_loss{val/loss:.6f}",
    )

    callbacks = [TQDMProgressBar(refresh_rate=20), ckpt]

    if bool(exp_conf.get("enable_early_stop", True)):
        callbacks.append(EarlyStopping(
            monitor="val/loss",
            patience=int(exp_conf.get("early_stop_patience", 10)),
            mode="min",
            verbose=True,
        ))

    trainer = Trainer(
        max_epochs=int(exp_conf["max_epochs"]),
        accelerator="auto",
        devices=training_conf["devices"],
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=float(exp_conf.get("gradient_clip_val", 1.0)),
    )

    trainer.fit(runner, datamodule=data_module)
    trainer.test(runner, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-s", "--save_dir", type=str, default="save")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("config", args.config)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    train_func(
        {"save_dir": args.save_dir, "devices": args.devices, "seed": args.seed},
        cfg.exp_conf,
    )
