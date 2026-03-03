# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch
import torch.nn as nn
import inspect
import pandas as pd
import matplotlib.pyplot as plt
from lightning.pytorch import Trainer, seed_everything, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

# パス設定
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from easytsf.runner.data_runner import NPYDataInterface


# 1. 重み付き損失関数 (Peak-Weighted MSE)
class PeakWeightedMSE(nn.Module):
    def __init__(self, peak_threshold=1.0, weight=10.0):
        super().__init__()
        self.peak_threshold = peak_threshold
        self.weight = weight

    def forward(self, pred, target):
        loss = (pred - target) ** 2
        weights = torch.ones_like(target)
        weights[target > self.peak_threshold] = self.weight
        return (loss * weights).mean()


def _apply_ghi_boost(var_x: torch.Tensor, ghi_index: int = 2, factor: float = 1.3) -> torch.Tensor:
    """
    GHIをブーストする（比較一貫性のため、train/val/testで同じ処理にする）
    """
    x = var_x.clone()
    if x.shape[-1] > ghi_index:
        x[:, :, ghi_index] = x[:, :, ghi_index] * factor
    return x


def _extract_pv_pred(pred: torch.Tensor, pv_index_if_multi: int = 3) -> torch.Tensor:
    """
    予測テンソル pred から「PV1本」を取り出す。

    - pred が (B, T, 1) のとき：そのまま返す（24h fromseries想定）
    - pred が (B, T, C>=2) のとき：従来互換で index=3 を優先（96h旧系想定）
      ただし C <= pv_index_if_multi のときは index=0 を使う（安全策）
    """
    if pred.dim() != 3:
        raise ValueError(f"[ERROR] pred の次元が想定外です: {pred.shape}")

    C = pred.shape[-1]
    if C == 1:
        return pred  # (B, T, 1)
    if C > pv_index_if_multi:
        return pred[:, :, pv_index_if_multi:pv_index_if_multi + 1]
    # フォールバック（Cが小さい場合）
    return pred[:, :, 0:1]


# 2. 学習実行クラス (PeakRunner)
class PeakRunner(LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.config = config
        self.loss_fn = PeakWeightedMSE(peak_threshold=1.0, weight=10.0)

        # 互換性のため：config側で上書きできるようにする
        self.ghi_index = int(self.config.get("ghi_index", 2))
        self.ghi_boost = float(self.config.get("ghi_boost", 1.3))
        self.pv_index = int(self.config.get("pv_index", 3))

    def forward(self, x_enc, x_mark_enc):
        return self.model(x_enc, x_mark_enc)

    def _predict(self, var_x, marker_x):
        # train/val/test 全部で同じ入力処理にする
        var_x_input = _apply_ghi_boost(var_x, ghi_index=self.ghi_index, factor=self.ghi_boost)
        outputs = self.model(var_x_input, marker_x)
        pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        pred_pv = _extract_pv_pred(pred, pv_index_if_multi=self.pv_index)
        return pred_pv

    def training_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred_pv = self._predict(var_x, marker_x)

        loss = self.loss_fn(pred_pv, var_y)

        if hasattr(self.model, 'get_load_balancing_loss'):
            loss += 0.1 * self.model.get_load_balancing_loss()

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred_pv = self._predict(var_x, marker_x)

        loss = nn.MSELoss()(pred_pv, var_y)
        mae = torch.nn.functional.l1_loss(pred_pv, var_y)

        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log("val/mae", mae, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred_pv = self._predict(var_x, marker_x)

        mse = torch.nn.functional.mse_loss(pred_pv, var_y)
        mae = torch.nn.functional.l1_loss(pred_pv, var_y)

        self.log("test/mse", mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/mae", mae, on_step=False, on_epoch=True, prog_bar=True)
        return mse

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.config.get("lr", 1e-4)))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
        return [optimizer], [scheduler]


def get_model(model_name, exp_conf):
    if model_name in ["iTransformerPeak", "iTransformer"]:
        from easytsf.model.iTransformer_peak import iTransformerPeak
        model_class = iTransformerPeak
    elif model_name == "MMK_Mix":
        from easytsf.model.MMK_Mix import MMK_Mix
        model_class = MMK_Mix
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
    model_runner = PeakRunner(base_model, exp_conf)
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
        gradient_clip_val=float(exp_conf.get("gradient_clip_val", 1.0))
    )

    trainer.fit(model=model_runner, datamodule=data_module)

    print("\n" + "=" * 30)
    print("  TEST EVALUATION (Best Model)")
    print("=" * 30)
    trainer.test(model=model_runner, datamodule=data_module, ckpt_path="best")

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

    train_func({"save_dir": args.save_dir, "devices": args.devices, "seed": args.seed}, config_module.exp_conf)
