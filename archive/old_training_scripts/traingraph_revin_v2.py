# -*- coding: utf-8 -*-
"""
traingraph_revin_v2.py

目的:
- RevIN + MMK_Mix_RevIN_bn の学習を「ピーク過小」を起こしにくい形に変更する
- 具体的には:
  (1) 夜間は marker で loss をほぼ見ない（夜の大量0に引っ張られない）
  (2) 昼間は出力が大きいほど重みを上げる（高ピークを学ぶ圧を強くする）
  (3) Huber系の安定した損失（外れ値に強い） + 相対誤差項（スケール依存を減らす）
"""

import os
import sys
import argparse
import inspect
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning.pytorch import Trainer, seed_everything, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

# パス設定
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from easytsf.runner.data_runner import NPYDataInterface


# -------------------------
# 1) 損失関数（ピーク対策）
# -------------------------
class DayPeakHuberLoss(nn.Module):
    """
    - marker_y を用いて「昼だけ」主に学習する
    - 出力が大きいほど重みを上げる（ピーク過小を防ぐ）
    - Huber + relative(相対誤差) を混ぜる

    期待するshape:
      pred_pv: (B, T)
      true_y : (B, T)
      marker : (B, T)  ※昼=1, 夜=0

    パラメータの意味（実務的に効く順）:
      peak_th : これ以上を「高出力」とみなす閾値（pv_kwhのスケールで決める）
      peak_w  : 高出力領域の重み倍率（5〜20くらいで調整）
      night_w : 夜の損失の重み（0〜0.1くらい、夜に引っ張られないように小さく）
      rel_w   : 相対誤差項の重み（0.1〜0.5くらいが無難）
      huber_delta : Huberのdelta（10〜30くらいが多い）
    """
    def __init__(self, peak_th=30.0, peak_w=10.0, night_w=0.02, rel_w=0.2, huber_delta=20.0, eps=1e-3):
        super().__init__()
        self.peak_th = float(peak_th)
        self.peak_w = float(peak_w)
        self.night_w = float(night_w)
        self.rel_w = float(rel_w)
        self.huber_delta = float(huber_delta)
        self.eps = float(eps)

    def forward(self, pred_pv, true_y, marker_y):
        # shape整形
        if pred_pv.dim() == 3:
            pred_pv = pred_pv.squeeze(-1)
        if true_y.dim() == 3:
            true_y = true_y.squeeze(-1)
        if marker_y.dim() == 3:
            marker_y = marker_y.squeeze(-1)

        pred_pv = pred_pv.float()
        true_y = true_y.float()
        marker_y = marker_y.float()

        # 非負（学習時も入れると安定することが多い）
        pred_pv = torch.clamp(pred_pv, min=0.0)

        # Huber（点ごと）
        hub = F.smooth_l1_loss(pred_pv, true_y, beta=self.huber_delta, reduction="none")  # (B,T)

        # 高出力ほど重みを上げる（true基準）
        peak_mask = (true_y >= self.peak_th).float()
        w_day = 1.0 + peak_mask * (self.peak_w - 1.0)  # 高出力は peak_w 倍

        # 昼・夜マスク
        day = marker_y
        night = 1.0 - marker_y

        # 昼損失（ピーク重視）
        loss_day = (hub * w_day * day).sum() / (day.sum() + self.eps)

        # 夜損失（弱め。夜の大量0が学習を支配しないように）
        loss_night = (hub * night).sum() / (night.sum() + self.eps)

        # 相対誤差（高出力に対して「割合」で当てにいく）
        rel = ((pred_pv - true_y).abs() / (true_y.abs() + 1.0)).clamp(max=10.0)  # (B,T)
        rel_day = (rel * w_day * day).sum() / (day.sum() + self.eps)

        loss = loss_day + self.night_w * loss_night + self.rel_w * rel_day
        return loss


def _apply_ghi_boost(var_x: torch.Tensor, ghi_index: int = 3, factor: float = 1.0) -> torch.Tensor:
    """
    既存互換：必要ならGHIを倍率で強調。
    ※ 今回は「過大な癖」を作りやすいのでデフォルト 1.0（無効）にするのを推奨
    """
    x = var_x.clone()
    if x.shape[-1] > ghi_index:
        x[:, :, ghi_index] = x[:, :, ghi_index] * factor
    return x


def _extract_pv_pred(pred: torch.Tensor, pv_index_if_multi: int = 0) -> torch.Tensor:
    """
    pred: (B, T, C)
    pvを1本取り出して (B, T) を返す
    """
    if pred.dim() != 3:
        raise ValueError(f"[ERROR] pred の次元が想定外: {pred.shape}")
    C = pred.shape[-1]
    if C == 1:
        return pred[:, :, 0]
    if 0 <= pv_index_if_multi < C:
        return pred[:, :, pv_index_if_multi]
    return pred[:, :, 0]


# -------------------------
# 2) Lightning Runner
# -------------------------
class PeakRunner(LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.config = config

        # index類（meta.jsonに合わせて設定済みの想定）
        self.pv_index = int(self.config.get("pv_index", 0))
        self.ghi_index = int(self.config.get("ghi_index", 3))
        self.ghi_boost = float(self.config.get("ghi_boost", 1.0))  # ★デフォ 1.0 推奨

        # 損失（ピーク対策）
        self.loss_fn = DayPeakHuberLoss(
            peak_th=float(self.config.get("peak_th", 30.0)),
            peak_w=float(self.config.get("peak_w", 10.0)),
            night_w=float(self.config.get("night_w", 0.02)),
            rel_w=float(self.config.get("rel_w", 0.2)),
            huber_delta=float(self.config.get("huber_delta", 20.0)),
        )

    def forward(self, x_enc, x_mark_enc):
        return self.model(x_enc, x_mark_enc)

    def _predict_pv(self, var_x, marker_x):
        var_x_input = _apply_ghi_boost(var_x, ghi_index=self.ghi_index, factor=self.ghi_boost)
        outputs = self.model(var_x_input, marker_x)
        pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        pred_pv = _extract_pv_pred(pred, pv_index_if_multi=self.pv_index)  # (B,T)
        return pred_pv

    def training_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred_pv = self._predict_pv(var_x, marker_x)  # (B,T)

        # ★marker_y を使って昼中心 + ピーク重視
        loss = self.loss_fn(pred_pv, var_y.squeeze(-1) if var_y.dim() == 3 else var_y,
                            marker_y.squeeze(-1) if marker_y.dim() == 3 else marker_y)

        # load balancing loss（MoKの場合）
        if hasattr(self.model, 'get_load_balancing_loss'):
            loss = loss + float(self.config.get("lb_w", 0.05)) * self.model.get_load_balancing_loss()

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred_pv = self._predict_pv(var_x, marker_x)

        # 評価指標は従来どおり（比較がしやすい）
        y = var_y.squeeze(-1) if var_y.dim() == 3 else var_y
        loss = F.mse_loss(pred_pv, y)
        mae = F.l1_loss(pred_pv, y)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/mae", mae, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred_pv = self._predict_pv(var_x, marker_x)

        y = var_y.squeeze(-1) if var_y.dim() == 3 else var_y
        mse = F.mse_loss(pred_pv, y)
        mae = F.l1_loss(pred_pv, y)

        self.log("test/mse", mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        return mse

    def configure_optimizers(self):
        lr = float(self.config.get("lr", 1e-4))
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=float(self.config.get("weight_decay", 1e-3)))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(self.config.get("step_size", 10)),
                                                    gamma=float(self.config.get("gamma", 0.5)))
        return [optimizer], [scheduler]


def get_model(model_name, exp_conf):
    if model_name == "MMK_Mix":
        from easytsf.model.MMK_Mix import MMK_Mix
        model_class = MMK_Mix
    elif model_name == "MMK_Mix_RevIN_bn":
        from easytsf.model.MMK_Mix_RevIN_bn import MMK_Mix_RevIN_bn
        model_class = MMK_Mix_RevIN_bn
    else:
        raise ValueError(f"Model {model_name} not supported.")

    sig = inspect.signature(model_class.__init__)
    valid_params = [p.name for p in sig.parameters.values() if p.name != 'self']
    filtered_conf = {k: v for k, v in exp_conf.items() if k in valid_params}
    return model_class(**filtered_conf)


def save_plots(log_dir):
    metrics_path = os.path.join(log_dir, "metrics.csv")
    if not os.path.exists(metrics_path):
        return
    df = pd.read_csv(metrics_path)

    plt.figure()
    if 'train/loss_epoch' in df.columns:
        plt.plot(df['train/loss_epoch'].dropna(), label='Train Loss')
    if 'val/loss' in df.columns:
        plt.plot(df['val/loss'].dropna(), label='Val Loss')
    plt.legend()
    plt.title("Loss Epoch")
    plt.savefig(os.path.join(log_dir, "loss_epoch.png"))
    print("[INFO] loss-epoch グラフを保存しました。")

    plt.figure()
    if 'val/mae' in df.columns:
        plt.plot(df['val/mae'].dropna(), label='Val MAE')
    plt.legend()
    plt.title("MAE Epoch")
    plt.savefig(os.path.join(log_dir, "mae_epoch.png"))
    print("[INFO] MAE-epoch グラフを保存しました。")


def train_func(training_conf, exp_conf):
    seed_everything(training_conf["seed"])

    base_model = get_model(exp_conf["model_name"], exp_conf)
    runner = PeakRunner(base_model, exp_conf)
    data_module = NPYDataInterface(**exp_conf)

    logger = CSVLogger(save_dir=training_conf["save_dir"], name=exp_conf["model_name"])

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="best-epoch{epoch:02d}-val_loss{val/loss:.4f}",
        save_top_k=5, mode="min", save_last=True
    )

    trainer = Trainer(
        max_epochs=int(exp_conf["max_epochs"]),
        accelerator="auto",
        devices=training_conf["devices"],
        logger=logger,
        callbacks=[
            TQDMProgressBar(refresh_rate=20),
            checkpoint_callback,
            EarlyStopping(monitor="val/loss", patience=int(exp_conf["early_stop_patience"]), mode="min", verbose=True)
        ],
        gradient_clip_val=float(exp_conf.get("gradient_clip_val", 1.0)),
        log_every_n_steps=int(exp_conf.get("log_every_n_steps", 10)),  # ★13バッチなら10くらいが見やすい
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
    parser.add_argument("-s", "--save_dir", type=str, default="save")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("config", args.config)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    train_func({"save_dir": args.save_dir, "devices": args.devices, "seed": args.seed}, config_module.exp_conf)
