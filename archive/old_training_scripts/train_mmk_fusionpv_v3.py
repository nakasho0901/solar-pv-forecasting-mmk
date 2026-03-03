# -*- coding: utf-8 -*-
import os
import sys
import argparse
import inspect
import importlib
from typing import Optional, List

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

from lightning.pytorch import Trainer, seed_everything, LightningModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar, Callback
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_info

# ============================================================
# パス設定（このファイルがあるプロジェクト直下を import できるように）
# ============================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from easytsf.runner.data_runner import NPYDataInterface
from loss_peak_weighted_v2 import PeakWeightedAsymMSE, PeakSpec


def filter_kwargs_for_callable(func, kwargs: dict) -> dict:
    """func の signature を見て、受け取れる kwargs だけ残す"""
    sig = inspect.signature(func)
    params = sig.parameters

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


def _apply_ghi_boost(var_x: torch.Tensor, ghi_index: int = 2, factor: float = 1.0) -> torch.Tensor:
    """入力側で GHI をブースト"""
    x = var_x.clone()
    if x.shape[-1] > ghi_index:
        x[:, :, ghi_index] = x[:, :, ghi_index] * factor
    return x


def _extract_ghi_proxy_for_loss(var_x: torch.Tensor, pred_len: int, ghi_weight_index: int) -> Optional[torch.Tensor]:
    """
    loss の重み用に GHI（proxy）を取り出す。
    ここでは「入力の最後 pred_len 分の GHI」を使う。
    ※未来GHIが無い設定のため、これは proxy に過ぎない。
    """
    if var_x is None or var_x.dim() != 3:
        return None
    if var_x.shape[-1] <= ghi_weight_index:
        return None
    return var_x[:, -pred_len:, ghi_weight_index]  # (B, pred_len)


def _extract_day_mask(marker_y: torch.Tensor) -> Optional[torch.Tensor]:
    """
    marker_y から daylight mask を取り出す。
    想定: marker_y (B, pred_len, 1) で is_daylight が入る。
    """
    if marker_y is None:
        return None
    if marker_y.dim() == 3:
        return marker_y[:, :, 0]
    if marker_y.dim() == 2:
        return marker_y
    return None


# ============================================================
# ★追加：ベスト更新時だけ見やすいログを出す Callback
# ============================================================
class BestOnlyConsoleLogger(Callback):
    """
    - monitor（例: "val/mae"）が best 更新したときだけ 1行出す
    - report_epochs（例: [0,1,4,7,15]）の epoch では、
      best更新が無くても「現時点のbest」を必ず 1行出す

    ※保存や学習挙動は一切変えない（printだけ）
    """

    def __init__(
        self,
        monitor: str = "val/mae",
        mode: str = "min",
        report_epochs: Optional[List[int]] = None,
        digits: int = 4,
    ):
        self.monitor = monitor
        self.mode = mode.lower().strip()
        self.report_epochs = report_epochs if report_epochs is not None else [0, 1, 4, 7, 15]
        self.digits = int(digits)

        if self.mode not in ("min", "max"):
            raise ValueError("[ERROR] mode must be 'min' or 'max'.")

        self.best = None  # type: Optional[float]
        self.best_epoch = None  # type: Optional[int]

    @staticmethod
    def _to_float(x) -> Optional[float]:
        if x is None:
            return None
        try:
            if isinstance(x, torch.Tensor):
                return float(x.detach().cpu().item())
            return float(x)
        except Exception:
            return None

    def _is_improved(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return current < self.best
        return current > self.best

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # Lightning が集計したメトリクス（epoch末）
        metrics = trainer.callback_metrics

        current_raw = metrics.get(self.monitor, None)
        current = self._to_float(current_raw)

        epoch = int(trainer.current_epoch)

        # best更新判定
        updated = False
        if current is not None and self._is_improved(current):
            self.best = current
            self.best_epoch = epoch
            updated = True
            rank_zero_info(
                f"[BEST更新] Epoch {epoch:>2d}  BEST({self.monitor}) = {current:.{self.digits}f}"
            )

        # 指定epochでは、更新が無くても現時点bestを出す
        if epoch in self.report_epochs:
            if self.best is None:
                rank_zero_info(f"[BEST現在] Epoch {epoch:>2d}  BEST({self.monitor}) = (まだ未確定)")
            else:
                rank_zero_info(
                    f"[BEST現在] Epoch {epoch:>2d}  BEST({self.monitor}) = {self.best:.{self.digits}f} (best@{self.best_epoch})"
                )


class LitWrapper(LightningModule):
    def __init__(self, exp_conf: dict):
        super().__init__()
        self.save_hyperparameters(exp_conf)

        self.hist_len = int(exp_conf["hist_len"])
        self.pred_len = int(exp_conf["pred_len"])

        self.pv_index = int(exp_conf.get("pv_index", 0))
        self.ghi_index = int(exp_conf.get("ghi_index", 3))
        self.ghi_boost = float(exp_conf.get("ghi_boost", 1.0))

        # loss control
        self.use_two_stage_loss = bool(exp_conf.get("use_two_stage_loss", True))
        self.warmup_epochs = int(exp_conf.get("warmup_epochs", 5))
        self.huber_delta = float(exp_conf.get("huber_delta", 10.0))

        self.use_day_mask_in_loss = bool(exp_conf.get("use_day_mask_in_loss", True))

        # GHI weight in loss (proxy)
        self.use_ghi_loss_weight = bool(exp_conf.get("use_ghi_loss_weight", False))
        self.ghi_loss_alpha = float(exp_conf.get("ghi_loss_alpha", 0.0))
        self.ghi_weight_index = int(exp_conf.get("ghi_weight_index", self.ghi_index))
        self.ghi_clip_max = float(exp_conf.get("ghi_clip_max", 1.0))

        # model
        self.model = self._build_model(exp_conf)

        # losses
        self.huber = nn.SmoothL1Loss(beta=self.huber_delta)

        peak_mode = str(exp_conf.get("peak_mode", "fixed"))
        peak = PeakSpec(
            mode=("quantile" if peak_mode == "quantile" else "fixed"),
            peak_threshold=float(exp_conf.get("peak_threshold", 1.0)),
            peak_quantile=float(exp_conf.get("peak_quantile", 0.9)),
            min_threshold=float(exp_conf.get("peak_min_threshold", 0.0)),
        )

        self.peak_loss = PeakWeightedAsymMSE(
            peak=peak,
            peak_weight=float(exp_conf.get("peak_weight", 10.0)),
            under_weight=float(exp_conf.get("under_weight", 2.0)),
            over_weight=float(exp_conf.get("over_weight", 1.0)),
            use_day_mask=self.use_day_mask_in_loss,
            use_ghi_weight=self.use_ghi_loss_weight,
            ghi_alpha=self.ghi_loss_alpha,
            ghi_clip_max=self.ghi_clip_max,
        )

        # optimizer
        self.lr = float(exp_conf.get("lr", 1e-4))
        self.lr_step_size = int(exp_conf.get("lr_step_size", 15))
        self.lr_gamma = float(exp_conf.get("lr_gamma", 0.5))

    def _build_model(self, exp_conf: dict):
        """
        config で model_module / model_class を指定できるようにする。
        例:
          model_module="easytsf.model.MMK_FusionPV_FeatureToken_v2"
          model_class ="MMK_FusionPV_FeatureToken"
        """
        mod_name = exp_conf.get("model_module", None)
        cls_name = exp_conf.get("model_class", None)

        if mod_name is None or cls_name is None:
            raise ValueError("[ERROR] config must define 'model_module' and 'model_class'.")

        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)

        filtered = filter_kwargs_for_callable(cls.__init__, exp_conf)
        return cls(**filtered)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=self.lr_step_size, gamma=self.lr_gamma)
        return {"optimizer": opt, "lr_scheduler": sch}

    def _predict(self, var_x: torch.Tensor, marker_x: torch.Tensor, marker_y: Optional[torch.Tensor] = None):
        var_x_in = _apply_ghi_boost(var_x, ghi_index=self.ghi_index, factor=self.ghi_boost)

        # モデルの受け口差分を吸収
        try:
            pred = self.model(var_x_in, marker_x, marker_y=marker_y)
        except TypeError:
            pred = self.model(var_x_in, marker_x)

        return pred, var_x_in

    def training_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred, var_x_input = self._predict(var_x, marker_x, marker_y=marker_y)

        day_mask = _extract_day_mask(marker_y) if self.use_day_mask_in_loss else None

        if self.use_two_stage_loss and (self.current_epoch < self.warmup_epochs):
            loss = self.huber(pred, var_y)
        else:
            ghi_proxy = None
            if self.use_ghi_loss_weight and (self.ghi_loss_alpha > 0.0):
                ghi_proxy = _extract_ghi_proxy_for_loss(
                    var_x=var_x_input,
                    pred_len=self.pred_len,
                    ghi_weight_index=self.ghi_weight_index,
                )
            loss = self.peak_loss(pred, var_y, day_mask=day_mask, ghi=ghi_proxy)

        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred, var_x_input = self._predict(var_x, marker_x, marker_y=marker_y)

        day_mask = _extract_day_mask(marker_y) if self.use_day_mask_in_loss else None

        ghi_proxy = None
        if self.use_ghi_loss_weight and (self.ghi_loss_alpha > 0.0):
            ghi_proxy = _extract_ghi_proxy_for_loss(
                var_x=var_x_input,
                pred_len=self.pred_len,
                ghi_weight_index=self.ghi_weight_index,
            )
        loss = self.peak_loss(pred, var_y, day_mask=day_mask, ghi=ghi_proxy)

        mae = torch.mean(torch.abs(pred - var_y))
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss, "val_mae": mae}

    def test_step(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        pred, _ = self._predict(var_x, marker_x, marker_y=marker_y)

        mae = torch.mean(torch.abs(pred - var_y))
        mse = torch.mean((pred - var_y) ** 2)

        self.log("test/mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/mse", mse, prog_bar=True, on_step=False, on_epoch=True)
        return {"test_mae": mae, "test_mse": mse}


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

    # Lightning の metrics.csv は train/val/test の行が混在し NaN が大量に出るので、
    # epoch ごとに「最後の non-NaN」を拾ってプロットする（これで必ず線が出る）
    if "epoch" not in df.columns:
        return

    def _pick_column(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    def _plot_epoch_series(colname: str, fname: str, ylabel: str):
        # epochごとに最後の有効値（non-NaN）を取る
        s = df[["epoch", colname]].dropna().groupby("epoch").last()
        if len(s) == 0:
            return
        x = s.index.values
        y = s[colname].values

        plt.figure()
        plt.plot(x, y, marker="o")
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, fname))
        plt.close()

    # もともとの意図（列名揺れ吸収）を維持しつつ、確実に描けるようにする
    train_loss_col = _pick_column(["train/loss_epoch", "train/loss"])
    val_mae_col = _pick_column(["val/mae_epoch", "val/mae"])

    if train_loss_col is not None:
        _plot_epoch_series(train_loss_col, "loss_epoch.png", "loss")

    if val_mae_col is not None:
        _plot_epoch_series(val_mae_col, "mae_epoch.png", "mae")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="config .py path")
    parser.add_argument("-s", "--save_dir", default="save", help="save root dir")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--resume_ckpt", type=str, default="", help="resume from this checkpoint (last.ckpt etc)")
    parser.add_argument("--accelerator", type=str, default="auto", help="auto/cpu/gpu")

    args = parser.parse_args()
    seed_everything(args.seed, workers=True)

    conf_mod = load_config_module(args.config)
    if not hasattr(conf_mod, "exp_conf"):
        raise AttributeError("[ERROR] config module has no exp_conf.")
    exp = conf_mod.exp_conf

    logger = CSVLogger(save_dir=args.save_dir, name=str(exp.get("model_name", "model")))

    callbacks = [TQDMProgressBar(refresh_rate=1)]

    # チェックポイント保存は今まで通り（val/loss最小）
    ckpt_cb = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=1,
        filename="best-epoch{epoch:02d}-val_loss{val/loss:.4f}",
    )
    callbacks.append(ckpt_cb)

    # ★追加：見やすい BEST ログ（学習や保存挙動は変えない）
    # - best_monitor: exp_conf["best_monitor"] があればそれを使う（例: "val/mae"）
    # - report_epochs: exp_conf["best_report_epochs"] があればそれを使う（例: [0,1,4,7,15]）
    best_monitor = str(exp.get("best_monitor", "val/mae"))
    best_mode = str(exp.get("best_mode", "min"))
    report_epochs = exp.get("best_report_epochs", [0, 1, 4, 7, 15])
    callbacks.append(
        BestOnlyConsoleLogger(
            monitor=best_monitor,
            mode=best_mode,
            report_epochs=list(report_epochs) if isinstance(report_epochs, (list, tuple)) else [0, 1, 4, 7, 15],
            digits=int(exp.get("best_digits", 4)),
        )
    )

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
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=float(exp.get("gradient_clip_val", 0.0)),
        log_every_n_steps=int(exp.get("log_every_n_steps", 50)),
        enable_checkpointing=True,
    )

    ckpt_path = args.resume_ckpt.strip() or None

    trainer.fit(model, datamodule=dm, ckpt_path=ckpt_path)
    trainer.test(model, datamodule=dm, ckpt_path="best")

    metrics_csv = os.path.join(logger.log_dir, "metrics.csv")
    plot_metrics(metrics_csv, logger.log_dir)
    print("[INFO] loss-epoch / mae-epoch グラフを保存しました。")


if __name__ == "__main__":
    main()
