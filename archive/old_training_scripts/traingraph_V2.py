# -*- coding: utf-8 -*-
import os
import sys
import argparse
import inspect
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from lightning.pytorch import Trainer, seed_everything, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

# ============================================================
# パス設定（このファイルがあるプロジェクト直下を import できるように）
# ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

# ★重要：あなたの easytsf には easytsf/data が無いので runner の NPYDataInterface を使う
from easytsf.runner.data_runner import NPYDataInterface


# ============================================================
# util: モデル __init__ が受け取れるキーだけ残す
# ============================================================
def filter_kwargs_for_callable(func, kwargs: dict) -> dict:
    """
    func のシグネチャを見て、受け取れる kwargs だけ残す。
    - func が **kwargs を受け取る場合：そのまま全部OK
    - そうでない場合：signature にある引数名だけ通す
    """
    sig = inspect.signature(func)
    params = sig.parameters

    # **kwargs を受け取るならフィルタ不要
    for p in params.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return dict(kwargs)

    allowed = set()
    for name, p in params.items():
        if name == "self":
            continue
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            allowed.add(name)

    return {k: v for k, v in kwargs.items() if k in allowed}


# ============================================================
# 損失関数
# ============================================================
class PeakWeightedMSE(nn.Module):
    """
    Peak-Weighted MSE (+ optional GHI-weight)

    - target が peak_threshold を超える部分に weight を掛ける
    - 追加：ghi が渡された場合、ghi の大きさに応じて誤差重みを増やす
    - 2つの重みは乗算で統合（ピークかつ高GHIを最重視）
    """

    def __init__(
        self,
        peak_threshold: float = 1.0,
        weight: float = 10.0,
        use_ghi_weight: bool = False,
        ghi_alpha: float = 0.0,
        ghi_clip_max: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.peak_threshold = float(peak_threshold)
        self.weight = float(weight)

        self.use_ghi_weight = bool(use_ghi_weight)
        self.ghi_alpha = float(ghi_alpha)
        self.ghi_clip_max = float(ghi_clip_max)
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, ghi: torch.Tensor | None = None) -> torch.Tensor:
        loss = (pred - target) ** 2

        # (1) Peak重み
        weights = torch.ones_like(target)
        weights[target > self.peak_threshold] = self.weight

        # (2) GHI重み（任意）
        if self.use_ghi_weight and (ghi is not None):
            if ghi.dim() == 3:
                ghi_ = ghi.squeeze(-1)
            elif ghi.dim() == 2:
                ghi_ = ghi
            else:
                ghi_ = None

            if ghi_ is not None:
                # バッチ内最大で割って 0〜1 正規化
                gmax = torch.clamp(ghi_.amax(dim=1, keepdim=True), min=self.eps)  # (B,1)
                ghi_norm = torch.clamp(ghi_ / gmax, min=0.0, max=self.ghi_clip_max)  # (B,T)

                w_ghi = 1.0 + self.ghi_alpha * ghi_norm  # (B,T)
                if target.dim() == 3:
                    w_ghi = w_ghi.unsqueeze(-1)  # (B,T,1)

                weights = weights * w_ghi

        return (loss * weights).mean()


def _apply_ghi_boost(var_x: torch.Tensor, ghi_index: int = 2, factor: float = 1.0) -> torch.Tensor:
    """入力側で GHI をブースト"""
    x = var_x.clone()
    if x.shape[-1] > ghi_index:
        x[:, :, ghi_index] = x[:, :, ghi_index] * factor
    return x


def _extract_ghi_for_loss(var_x: torch.Tensor, pred_len: int, ghi_weight_index: int) -> torch.Tensor | None:
    """
    loss の重み用に GHI（または proxy）を取り出す。
    ここでは「入力の最後 pred_len 分の GHI」を使う。
    """
    if var_x is None or var_x.dim() != 3:
        return None
    if var_x.shape[-1] <= ghi_weight_index:
        return None
    ghi = var_x[:, -pred_len:, ghi_weight_index]  # (B, pred_len)
    return ghi


# ============================================================
# LightningModule（学習ループ）
# ============================================================
class LitWrapper(LightningModule):
    def __init__(self, exp_conf: dict):
        super().__init__()
        self.save_hyperparameters(exp_conf)

        self.model_name = exp_conf["model_name"]
        self.hist_len = int(exp_conf["hist_len"])
        self.pred_len = int(exp_conf["pred_len"])

        # indices / boosts
        self.pv_index = int(exp_conf.get("pv_index", 0))
        self.ghi_index = int(exp_conf.get("ghi_index", 3))
        self.ghi_boost = float(exp_conf.get("ghi_boost", 1.0))

        # loss control
        self.use_two_stage_loss = bool(exp_conf.get("use_two_stage_loss", True))
        self.warmup_epochs = int(exp_conf.get("warmup_epochs", 5))
        self.huber_delta = float(exp_conf.get("huber_delta", 10.0))

        self.use_ghi_loss_weight = bool(exp_conf.get("use_ghi_loss_weight", False))
        self.ghi_loss_alpha = float(exp_conf.get("ghi_loss_alpha", 0.0))
        self.ghi_weight_index = int(exp_conf.get("ghi_weight_index", self.ghi_index))
        self.ghi_clip_max = float(exp_conf.get("ghi_clip_max", 1.0))

        # night constraint（ここでは使用フラグだけ保持）
        self.force_night0 = bool(exp_conf.get("force_night0", False))

        # model
        self.model = self._build_model(exp_conf)

        self.huber = nn.SmoothL1Loss(beta=self.huber_delta)
        self.peak_mse = PeakWeightedMSE(
            peak_threshold=float(exp_conf.get("peak_threshold", 1.0)),
            weight=float(exp_conf.get("peak_weight", 10.0)),
            use_ghi_weight=self.use_ghi_loss_weight,
            ghi_alpha=self.ghi_loss_alpha,
            ghi_clip_max=self.ghi_clip_max,
        )

        self.lr = float(exp_conf.get("lr", 1e-4))
        self.lr_step_size = int(exp_conf.get("lr_step_size", 15))
        self.lr_gamma = float(exp_conf.get("lr_gamma", 0.5))

    def _build_model(self, exp_conf: dict):
        model_name = exp_conf["model_name"]

        if model_name == "MMK_FusionPV_FeatureToken":
            from easytsf.model.MMK_FusionPV_FeatureToken import MMK_FusionPV_FeatureToken
            filtered = filter_kwargs_for_callable(MMK_FusionPV_FeatureToken.__init__, exp_conf)
            return MMK_FusionPV_FeatureToken(**filtered)

        if model_name == "MMK_Mix":
            from easytsf.model.MMK_Mix import MMK_Mix
            filtered = filter_kwargs_for_callable(MMK_Mix.__init__, exp_conf)
            return MMK_Mix(**filtered)

        raise ValueError(f"[ERROR] model_name not supported in traingraph_V2.py: {model_name}")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=self.lr_step_size, gamma=self.lr_gamma)
        return {"optimizer": opt, "lr_scheduler": sch}

    def _predict(self, var_x: torch.Tensor, marker_x: torch.Tensor, marker_y: torch.Tensor | None = None):
        var_x_in = _apply_ghi_boost(var_x, ghi_index=self.ghi_index, factor=self.ghi_boost)

        # モデルの受け口差分を吸収
        try:
            pred = self.model(var_x_in, marker_x, marker_y=marker_y)
        except TypeError:
            pred = self.model(var_x_in, marker_x)

        return pred, var_x_in

    def training_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred_pv, var_x_input = self._predict(var_x, marker_x, marker_y=marker_y)

        if self.use_two_stage_loss and (self.current_epoch < self.warmup_epochs):
            loss = self.huber(pred_pv, var_y)
        else:
            ghi_for_loss = None
            if self.use_ghi_loss_weight and (self.ghi_loss_alpha > 0.0):
                ghi_for_loss = _extract_ghi_for_loss(
                    var_x=var_x_input,
                    pred_len=self.pred_len,
                    ghi_weight_index=self.ghi_weight_index,
                )
            loss = self.peak_mse(pred_pv, var_y, ghi=ghi_for_loss)

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred_pv, var_x_input = self._predict(var_x, marker_x, marker_y=marker_y)

        ghi_for_loss = None
        if self.use_ghi_loss_weight and (self.ghi_loss_alpha > 0.0):
            ghi_for_loss = _extract_ghi_for_loss(
                var_x=var_x_input,
                pred_len=self.pred_len,
                ghi_weight_index=self.ghi_weight_index,
            )
        loss = self.peak_mse(pred_pv, var_y, ghi=ghi_for_loss)

        mae = torch.mean(torch.abs(pred_pv - var_y))
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_mae": mae}

    def test_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred_pv, _ = self._predict(var_x, marker_x, marker_y=marker_y)

        mae = torch.mean(torch.abs(pred_pv - var_y))
        mse = torch.mean((pred_pv - var_y) ** 2)

        self.log("test/mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/mse", mse, prog_bar=True, on_step=False, on_epoch=True)
        return {"test_mae": mae, "test_mse": mse}


# ============================================================
# main
# ============================================================
def load_config_module(path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("exp_config", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def plot_metrics(csv_path: str, outdir: str):
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    os.makedirs(outdir, exist_ok=True)

    cols = df.columns.tolist()

    def _plot(colname: str, fname: str, ylabel: str):
        if colname not in df.columns:
            return
        x = list(range(len(df[colname].values)))
        plt.figure()
        plt.plot(x, df[colname].values)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname))
        plt.close()

    for c in ["train/loss_epoch", "train/loss"]:
        if c in cols:
            _plot(c, "loss_epoch.png", "loss")
            break
    for c in ["val/mae", "val/mae_epoch"]:
        if c in cols:
            _plot(c, "mae_epoch.png", "mae")
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="config .py path")
    parser.add_argument("-s", "--save_dir", default="save", help="save root dir")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--resume_ckpt", type=str, default="", help="resume from this checkpoint (last.ckpt etc)")

    args = parser.parse_args()
    seed_everything(args.seed, workers=True)

    conf_mod = load_config_module(args.config)
    if not hasattr(conf_mod, "exp_conf"):
        raise AttributeError("[ERROR] config module has no exp_conf. Check the config file contents.")
    exp = conf_mod.exp_conf

    logger = CSVLogger(save_dir=args.save_dir, name=exp["model_name"])

    callbacks = [TQDMProgressBar(refresh_rate=1)]
    ckpt_cb = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=1,
        filename="best-epoch{epoch:02d}-val_loss{val/loss:.4f}",
    )
    callbacks.append(ckpt_cb)

    if exp.get("enable_early_stop", True):
        callbacks.append(
            EarlyStopping(
                monitor="val/loss",
                mode="min",
                patience=int(exp.get("early_stop_patience", 10)),
                min_delta=float(exp.get("early_stop_min_delta", 0.0)),
            )
        )

    dm = NPYDataInterface(
        data_dir=exp["data_dir"],
        batch_size=int(exp.get("batch_size", 64)),
        num_workers=int(exp.get("num_workers", 0)),
    )

    model = LitWrapper(exp)

    trainer = Trainer(
        max_epochs=int(exp.get("max_epochs", 30)),
        accelerator="cpu",
        devices=args.devices,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=float(exp.get("gradient_clip_val", 0.0)),
        log_every_n_steps=int(exp.get("log_every_n_steps", 50)),
        enable_checkpointing=True,
    )

    ckpt_path = args.resume_ckpt.strip()
    if ckpt_path == "":
        ckpt_path = None

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    trainer.test(model, datamodule=dm, ckpt_path="best")

    metrics_csv = os.path.join(logger.log_dir, "metrics.csv")
    plot_metrics(metrics_csv, logger.log_dir)
    print("[INFO] loss-epoch / mae-epoch グラフを保存しました。")


if __name__ == "__main__":
    main()
