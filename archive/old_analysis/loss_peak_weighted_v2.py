# -*- coding: utf-8 -*-
r"""
loss_peak_weighted_v2.py

目的:
- 「ピークを追いすぎて曇雨で過大になる」副作用を抑えつつ、
  「ピーク不足(pred < true)」だけを強く罰してピークを戻す。

特徴:
- Peak定義:
  - fixed閾値 (peak_threshold)
  - または quantile (peak_quantile) で上位p%だけをピーク扱い
- Asymmetric:
  - under (pred<true) に強い重み
  - over  (pred>true) は弱め/通常
- day_mask:
  - daylight のみ重みを掛けたい時に使う（夜を混ぜると評価が壊れる）
- ghi_weight:
  - 将来GHIが無いので、入力末尾から作る proxy を使う想定（任意）
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Literal

import torch
import torch.nn as nn


@dataclass
class PeakSpec:
    mode: Literal["fixed", "quantile"] = "fixed"
    peak_threshold: float = 1.0          # fixed のとき使用（単位: pv_kwh）
    peak_quantile: float = 0.9           # quantile のとき使用（上位10%など）
    min_threshold: float = 0.0           # quantile 算出後の下限（安全弁）


class PeakWeightedAsymMSE(nn.Module):
    """
    Peak-weighted + Asymmetric MSE

    total_weight = base * peak_weight * asym_weight * (optional ghi_weight)

    - peak_weight: peak対象だけ掛ける（昼の全点に掛けないのが重要）
    - asym_weight: underだけ強く罰する（曇雨の過大を増やしにくい）
    """

    def __init__(
        self,
        peak: PeakSpec,
        peak_weight: float = 10.0,
        under_weight: float = 2.0,
        over_weight: float = 1.0,
        use_day_mask: bool = True,
        # GHI weighting
        use_ghi_weight: bool = False,
        ghi_alpha: float = 0.0,
        ghi_clip_max: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.peak = peak
        self.peak_weight = float(peak_weight)
        self.under_weight = float(under_weight)
        self.over_weight = float(over_weight)
        self.use_day_mask = bool(use_day_mask)

        self.use_ghi_weight = bool(use_ghi_weight)
        self.ghi_alpha = float(ghi_alpha)
        self.ghi_clip_max = float(ghi_clip_max)
        self.eps = float(eps)

    @torch.no_grad()
    def _compute_peak_threshold(self, target: torch.Tensor, day_mask: Optional[torch.Tensor]) -> float:
        """
        target: (B,T) or (B,T,1)
        day_mask: (B,T) bool/0-1  (optional)
        """
        y = target
        if y.dim() == 3:
            y = y.squeeze(-1)  # (B,T)

        if day_mask is not None:
            m = day_mask
            if m.dtype != torch.bool:
                m = m > 0.5
            y = y[m]  # flatten daylight only

        if y.numel() == 0:
            # fallback：何も無い場合は固定閾値
            return float(self.peak.peak_threshold)

        if self.peak.mode == "fixed":
            th = float(self.peak.peak_threshold)
        else:
            q = float(self.peak.peak_quantile)
            q = max(0.0, min(1.0, q))
            th = float(torch.quantile(y.float(), q).item())

        th = max(th, float(self.peak.min_threshold))
        return th

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        day_mask: Optional[torch.Tensor] = None,
        ghi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        pred/target: (B,T,1) or (B,T)
        day_mask:    (B,T) bool/float (1=day)
        ghi:         (B,T) or (B,T,1)  (optional)
        """
        # shapes
        if pred.dim() == 2:
            pred_ = pred.unsqueeze(-1)
        else:
            pred_ = pred
        if target.dim() == 2:
            target_ = target.unsqueeze(-1)
        else:
            target_ = target

        # base mse
        err = pred_ - target_
        loss = err ** 2  # (B,T,1)

        # prepare mask (daylight)
        dm = day_mask
        if dm is not None and dm.dim() == 3:
            dm = dm.squeeze(-1)
        if self.use_day_mask and dm is not None:
            # night is weight 0 (do not contribute)
            dm_w = dm.float().clamp(0.0, 1.0).unsqueeze(-1)  # (B,T,1)
        else:
            dm_w = torch.ones_like(loss)

        # peak threshold (computed on target, daylight only)
        th = self._compute_peak_threshold(target_, dm)

        # peak weight
        peak_w = torch.ones_like(loss)
        peak_w[target_ > th] = self.peak_weight

        # asymmetric weight (under/over)
        asym_w = torch.ones_like(loss)
        asym_w[err < 0] = self.under_weight  # under-prediction
        asym_w[err > 0] = self.over_weight   # over-prediction

        # optional ghi weight
        ghi_w = torch.ones_like(loss)
        if self.use_ghi_weight and (ghi is not None) and (self.ghi_alpha > 0.0):
            g = ghi
            if g.dim() == 3:
                g = g.squeeze(-1)  # (B,T)
            # normalize per-sample
            gmax = torch.clamp(g.amax(dim=1, keepdim=True), min=self.eps)  # (B,1)
            gnorm = torch.clamp(g / gmax, min=0.0, max=self.ghi_clip_max)  # (B,T)
            ghi_w = (1.0 + self.ghi_alpha * gnorm).unsqueeze(-1)  # (B,T,1)

        w = dm_w * peak_w * asym_w * ghi_w
        # normalize by sum of weights (to keep scale stable)
        denom = torch.clamp(w.sum(), min=self.eps)
        return (loss * w).sum() / denom
