# easytsf/runner/exp_runner.py
# -*- coding: utf-8 -*-
"""
LTSFRunner（Long-Term Series Forecasting Runner）

- iTransformer 用
- 入力: (var_x, marker_x, var_y, marker_y)
    var_x    : [B, hist_len, enc_in]
    marker_x : [B, hist_len, t_in]  ※ iTransformer 実装では使わないが形は受け取る
    var_y    : [B, pred_len, 1]     ※ PV のみ（通常は標準化済み）
    marker_y : [B, pred_len, t_out] ※ hardzero では ghi_future:[B,pred_len,1] が入る

【hardzero 対応】
- marker_y が hardzero の ghi_future のとき：
  - mask = (marker_y > ghi_threshold) を作って「昼だけLoss」
  - 任意：夜は予測を 0[kW]（スケール空間の0）に強制
"""

import importlib
import inspect
import os

import lightning.pytorch as L
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrs


class LTSFRunner(L.LightningModule):
    def __init__(self, **kargs):
        super().__init__()
        self.save_hyperparameters()

        # ---------------------------------------------------------
        # 逆標準化用統計（存在しない場合はフェイルセーフ）
        # ---------------------------------------------------------
        try:
            stat_path = os.path.join(self.hparams.data_root, f"{self.hparams.dataset_name}.npz")
            stat = np.load(stat_path, allow_pickle=True)
            mean = torch.tensor(stat["mean"]).float()
            std = torch.tensor(stat["std"]).float()
        except Exception:
            mean = torch.tensor(0.0).float()
            std = torch.tensor(1.0).float()

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        # ---------------------------------------------------------
        # モデル構築 & 損失関数の設定
        # ---------------------------------------------------------
        self.load_model()
        self.configure_loss()

        # PV なので負の出力はカットするオプション
        self.enforce_nonneg = bool(getattr(self.hparams, "enforce_nonneg", True))

        # ---------------------------------------------------------
        # hardzero 設定
        # ---------------------------------------------------------
        # しきい値（GHI の単位は build_dataset_hardzero と同じ想定）
        self.ghi_threshold = float(getattr(self.hparams, "ghi_threshold", 1.0))
        # night を 0[kW] に強制するか（物理制約を強めたい場合 True）
        self.force_night0 = bool(getattr(self.hparams, "force_night0", True))
        # marker_y を使ってマスクLossするか（hardzero データなら True 推奨）
        self.use_marker_mask = bool(getattr(self.hparams, "use_marker_mask", True))

        # 0[kW] の「スケール空間の値」をキャッシュ（data_dir/scaler_y.npz があればそれ優先）
        self._zero_scaled = None

    # ---------------------------------------------------------
    # internal utils
    # ---------------------------------------------------------
    def _get_zero_scaled(self) -> torch.Tensor:
        """
        0[kW] を「y のスケール空間」に変換した値を返す。
        - data_dir/scaler_y.npz（mean,std）を最優先（今回の processed / processed_hardzero と整合）
        - 無い場合は buffer(mean,std) を使う
        """
        if self._zero_scaled is not None:
            return self._zero_scaled.to(self.device)

        z = None
        data_dir = getattr(self.hparams, "data_dir", None)

        if data_dir is not None:
            sc_path = os.path.join(data_dir, "scaler_y.npz")
            if os.path.exists(sc_path):
                sc = np.load(sc_path, allow_pickle=True)
                m = float(np.atleast_1d(sc["mean"])[0])
                s = float(np.atleast_1d(sc["std"])[0])
                z = (0.0 - m) / (s + 1e-12)

        if z is None:
            # fallback: buffer mean/std（形がスカラーであることが多い）
            m = self.mean
            s = self.std
            # mean/std が配列の場合は先頭要素を使う（PVは1ch想定）
            if m.numel() > 1:
                m = m.view(-1)[0]
            if s.numel() > 1:
                s = s.view(-1)[0]
            z = (0.0 - float(m.item())) / (float(s.item()) + 1e-12)

        self._zero_scaled = torch.tensor(z, dtype=torch.float32)
        return self._zero_scaled.to(self.device)

    def _is_hardzero_marker(self, marker_y: torch.Tensor) -> bool:
        """
        marker_y が hardzero の ghi_future っぽいか判定。
        - hardzero の marker_y は [B,T,1] で、値にばらつきがある（0一色ではない）想定
        - 通常の y_mark は 4次元時間特徴 or 0埋めなので、ここで弾く
        """
        if marker_y is None:
            return False
        if marker_y.dim() != 3:
            return False
        if marker_y.size(-1) != 1:
            return False
        # 全ゼロっぽいなら hardzero ではない（ダミー）
        if torch.isfinite(marker_y).all():
            if marker_y.abs().max().item() < 1e-8:
                return False
        return True

    def _masked_losses(self, pred: torch.Tensor, label: torch.Tensor, marker_y: torch.Tensor):
        """
        hardzero 用：
        - mask = marker_y > ghi_threshold（昼=1, 夜=0）
        - loss: 昼のみMSE
        - mae : 昼のみMAE
        - オプション：夜は pred を 0[kW] に強制
        """
        mask = (marker_y > self.ghi_threshold).float()  # [B,T,1]
        eps = 1e-8

        if self.force_night0:
            z = self._get_zero_scaled()  # scalar
            pred = pred * mask + z * (1.0 - mask)

        diff = pred - label
        mse_num = (diff ** 2 * mask).sum()
        mae_num = (diff.abs() * mask).sum()
        den = mask.sum() + eps

        loss_day = mse_num / den
        mae_day = mae_num / den

        # 参考：全体（夜も含む）
        loss_all = torch.mean(diff ** 2)
        mae_all = torch.mean(diff.abs())

        return pred, loss_day, mae_day, loss_all, mae_all, mask

    # ---------------------------------------------------------
    # forward
    # ---------------------------------------------------------
    def forward(self, batch, batch_idx):
        """
        DataModule から渡される batch:
            (var_x, marker_x, var_y, marker_y)
        から、モデル出力 pred と教師ラベル label を作る。
        """
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]

        try:
            out = self.model(var_x, marker_x)
        except TypeError:
            out = self.model(var_x)

        label = var_y[:, -self.hparams.pred_len:, :]

        if out.dim() == 3 and out.size(-1) > 1:
            t_idx = int(getattr(self.hparams, "target_col_idx", 0))
            pred = out[:, -self.hparams.pred_len:, t_idx:t_idx + 1]
        else:
            pred = out[:, -self.hparams.pred_len:, ...]
            if pred.dim() == 2:
                pred = pred.unsqueeze(-1)

        if self.enforce_nonneg:
            pred = torch.relu(pred)

        return pred, label, marker_y

    # ---------------------------------------------------------
    # Step 定義
    # ---------------------------------------------------------
    def training_step(self, batch, batch_idx):
        pred, label, marker_y = self.forward(batch, batch_idx)

        use_mask = self.use_marker_mask and self._is_hardzero_marker(marker_y)
        if use_mask:
            pred, loss_day, mae_day, loss_all, mae_all, mask = self._masked_losses(pred, label, marker_y)
            loss = loss_day
            mae = mae_day

            # 参考ログ（全体）
            self.log("train/loss_all", loss_all, on_step=True, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train/mae_all", mae_all, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train/day_ratio", mask.mean(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        else:
            loss = self.loss_function(pred, label)
            mae = torch.nn.functional.l1_loss(pred, label)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/mae", mae, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, label, marker_y = self.forward(batch, batch_idx)

        use_mask = self.use_marker_mask and self._is_hardzero_marker(marker_y)
        if use_mask:
            pred, loss_day, mae_day, loss_all, mae_all, mask = self._masked_losses(pred, label, marker_y)
            loss = loss_day
            mae = mae_day

            self.log("val/loss_all", loss_all, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("val/mae_all", mae_all, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("val/day_ratio", mask.mean(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        else:
            loss = self.loss_function(pred, label)
            mae = torch.nn.functional.l1_loss(pred, label)

        # EarlyStopping/Checkpoint は val/loss を監視している前提 → hardzeroなら昼lossになる
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mae", mae, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        pred, label, marker_y = self.forward(batch, batch_idx)

        use_mask = self.use_marker_mask and self._is_hardzero_marker(marker_y)
        if use_mask:
            pred, loss_day, mae_day, loss_all, mae_all, mask = self._masked_losses(pred, label, marker_y)

            # 互換：従来キーは "all" を入れておく（比較しやすい）
            self.log("test/mae", mae_all, on_step=False, on_epoch=True, sync_dist=True)
            self.log("test/mse", loss_all, on_step=False, on_epoch=True, sync_dist=True)

            # hardzeroで見たい指標（昼だけ）
            self.log("test/mae_day", mae_day, on_step=False, on_epoch=True, sync_dist=True)
            self.log("test/mse_day", loss_day, on_step=False, on_epoch=True, sync_dist=True)
            self.log("test/day_ratio", mask.mean(), on_step=False, on_epoch=True, sync_dist=True)
        else:
            mae = torch.nn.functional.l1_loss(pred, label)
            mse = torch.nn.functional.mse_loss(pred, label)
            self.log("test/mae", mae, on_step=False, on_epoch=True, sync_dist=True)
            self.log("test/mse", mse, on_step=False, on_epoch=True, sync_dist=True)

    # ---------------------------------------------------------
    # Loss・Optimizer
    # ---------------------------------------------------------
    def configure_loss(self):
        """損失関数の設定（今回は MSE）"""
        self.loss_function = nn.MSELoss()

    def configure_optimizers(self):
        """
        Optimizer と Scheduler の設定
        - Optimizer: AdamW
        - Scheduler: StepLR
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.optimizer_weight_decay,
            betas=(0.9, 0.95),
        )

        scheduler = lrs.StepLR(
            optimizer,
            step_size=self.hparams.lr_step_size,
            gamma=self.hparams.lr_gamma,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    # ---------------------------------------------------------
    # モデルロード
    # ---------------------------------------------------------
    def load_model(self):
        """easytsf.model.{model_name} からクラスを import してインスタンス化"""
        model_name = self.hparams.model_name
        module = importlib.import_module("." + model_name, package="easytsf.model")
        ModelClass = getattr(module, model_name)
        self.model = self.instancialize(ModelClass)

    def instancialize(self, cls):
        """__init__ のシグネチャを見て、hparams から引数を拾ってインスタンス化"""
        sig = inspect.signature(cls.__init__)
        kwargs = {}
        for k, p in sig.parameters.items():
            if k == "self":
                continue
            if hasattr(self.hparams, k):
                kwargs[k] = getattr(self.hparams, k)
        return cls(**kwargs)
