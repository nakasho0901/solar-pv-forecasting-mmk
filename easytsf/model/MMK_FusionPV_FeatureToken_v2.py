# -*- coding: utf-8 -*-
r"""
MMK_FusionPV_FeatureToken_v2.py

元: MMK_FusionPV_FeatureToken.py
目的: 新命名で固定し、今後の改良をこのv2系列で管理する。

機能:
- FeatureToken + Feature-wise gating (MoK)
- baseline: last24_repeat
- PV RevIN
- nonneg clamp
- return_gate=True で gate を返す
"""

import torch
import torch.nn as nn

from easytsf.layer.kanlayer import KANInterfaceV2


class PVRevIN(nn.Module):
    """PV専用 RevIN（B, L, 1 を時間方向に正規化）"""
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)

    def normalize(self, pv: torch.Tensor):
        mean = pv.mean(dim=1, keepdim=True)  # (B,1,1)
        var = pv.var(dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        pv_norm = (pv - mean) / std
        return pv_norm, mean, std

    def denormalize(self, y_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        return y_norm * std + mean


class FeatureTokenEncoder(nn.Module):
    """
    各特徴量 f の系列 x[:, :, f] (B, L) を小さなMLPで D 次元 token にする。
    入力:  (B, L, F)
    出力:  (B, F, D)
    """
    def __init__(self, hist_len: int, var_num: int, token_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.hist_len = int(hist_len)
        self.var_num = int(var_num)
        self.token_dim = int(token_dim)

        self.encoder = nn.Sequential(
            nn.Linear(self.hist_len, self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.token_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, F = x.shape
        assert L == self.hist_len, f"hist_len mismatch: got {L}, expected {self.hist_len}"
        assert F == self.var_num, f"var_num mismatch: got {F}, expected {self.var_num}"

        xf = x.permute(0, 2, 1).contiguous()  # (B,F,L)
        xf = xf.view(B * F, L)                # (B*F,L)
        tf = self.encoder(xf)                 # (B*F,D)
        tf = tf.view(B, F, self.token_dim)    # (B,F,D)
        return tf


class FeatureMoK(nn.Module):
    """
    特徴量tokenごとにゲートを出し、expertを混合して token を更新する。
    入力:  tokens (B, F, D)
    出力:  tokens' (B, F, D), gates (B, F, E)
    """
    def __init__(self, dim: int, layer_hp):
        super().__init__()
        self.dim = int(dim)

        experts = []
        if isinstance(layer_hp, (list, tuple)):
            if len(layer_hp) == 0:
                raise ValueError("[ERROR] layer_hp(list) is empty.")
            for spec in layer_hp:
                if not (isinstance(spec, (list, tuple)) and len(spec) == 2):
                    raise TypeError(f"[ERROR] layer_hp element must be [layer_type, hyperparam], got: {spec}")
                layer_type = str(spec[0])
                hyperparam = spec[1]
                experts.append(KANInterfaceV2(self.dim, self.dim, layer_type, hyperparam))
            self.n_expert = len(experts)

        elif isinstance(layer_hp, dict):
            n_expert = int(layer_hp["n_expert"])
            kan_hp = dict(layer_hp["kan_hp"])
            layer_type = str(kan_hp.get("layer_type", "KAN"))
            hyperparam = kan_hp.get("hyperparam", 5)
            for _ in range(n_expert):
                experts.append(KANInterfaceV2(self.dim, self.dim, layer_type, hyperparam))
            self.n_expert = n_expert

        else:
            raise TypeError(f"[ERROR] layer_hp must be list or dict, got: {type(layer_hp)}")

        self.experts = nn.ModuleList(experts)

        self.gate = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.n_expert),
        )

    def forward(self, tokens: torch.Tensor):
        B, F, D = tokens.shape
        assert D == self.dim

        t = tokens.reshape(B * F, D)                 # (B*F,D)
        logits = self.gate(t)                        # (B*F,E)
        scores = torch.softmax(logits, dim=-1).view(B, F, self.n_expert)  # (B,F,E)

        outs = [ex(t) for ex in self.experts]        # list[(B*F,D)]
        stacked = torch.stack(outs, dim=1)           # (B*F,E,D)
        w = scores.view(B * F, self.n_expert).unsqueeze(-1)  # (B*F,E,1)
        y = (stacked * w).sum(dim=1)                 # (B*F,D)
        y = y.view(B, F, D)                          # (B,F,D)
        return y, scores

    def export_expert_formulas(self):
        formulas = []
        for i, ex in enumerate(self.experts):
            info = {"expert_id": i, "class": ex.__class__.__name__}

            if hasattr(ex, "export_formula"):
                try:
                    info["formula"] = ex.export_formula()
                except Exception as e:
                    info["formula_error"] = str(e)
            elif hasattr(ex, "get_formula"):
                try:
                    info["formula"] = ex.get_formula()
                except Exception as e:
                    info["formula_error"] = str(e)
            elif hasattr(ex, "symbolic"):
                try:
                    info["formula"] = ex.symbolic()
                except Exception as e:
                    info["formula_error"] = str(e)

            if "formula" not in info:
                info["param_names"] = [n for (n, _) in ex.named_parameters()]

            formulas.append(info)
        return formulas


class FeatureMoKBlock(nn.Module):
    """FeatureMoK + Residual + Norm"""
    def __init__(self, dim: int, layer_hp, use_norm: bool = True, dropout: float = 0.1):
        super().__init__()
        self.mok = FeatureMoK(dim, layer_hp)
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor):
        y, gates = self.mok(tokens)
        tokens = tokens + y
        tokens = self.dropout(self.norm(tokens))
        return tokens, gates

    def export_expert_formulas(self):
        return self.mok.export_expert_formulas()


class MMK_FusionPV_FeatureToken(nn.Module):
    """
    FusionPV（特徴量別ゲート版）

    入力:
      var_x: (B, hist_len, var_num)
      marker_x: (B, hist_len, marker_dim)  ※未使用（特徴量にhour_sin/cos等を含める方針）
    出力:
      pred: (B, pred_len, 1)

    return_gate=True のとき (pred, gates_all)
      gates_all: list[Tensor]  各層の gate (B, F, E)
    """
    def __init__(
        self,
        hist_len: int,
        pred_len: int,
        var_num: int,
        pv_index: int = 0,

        token_dim: int = 64,
        layer_num: int = 3,
        layer_hp=None,
        use_layernorm: bool = True,
        dropout: float = 0.1,

        fusion: str = "mean",  # "mean" or "concat"
        baseline_mode: str = "last24_repeat",

        use_pv_revin: bool = True,
        revin_eps: float = 1e-5,

        enforce_nonneg: bool = True,
    ):
        super().__init__()
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.var_num = int(var_num)
        self.pv_index = int(pv_index)

        self.token_dim = int(token_dim)
        self.layer_num = int(layer_num)
        self.layer_hp = layer_hp
        self.use_layernorm = bool(use_layernorm)
        self.dropout = float(dropout)

        self.fusion = str(fusion)
        self.baseline_mode = str(baseline_mode)

        self.use_pv_revin = bool(use_pv_revin)
        self.pv_revin = PVRevIN(eps=revin_eps)

        self.enforce_nonneg = bool(enforce_nonneg)

        self.token_encoder = FeatureTokenEncoder(
            hist_len=self.hist_len,
            var_num=self.var_num,
            token_dim=self.token_dim,
            dropout=self.dropout,
        )

        self.blocks = nn.ModuleList([
            FeatureMoKBlock(self.token_dim, self.layer_hp, use_norm=self.use_layernorm, dropout=self.dropout)
            for _ in range(self.layer_num)
        ])

        if self.fusion == "mean":
            fused_dim = self.token_dim
        elif self.fusion == "concat":
            fused_dim = self.var_num * self.token_dim
        else:
            raise ValueError(f"[ERROR] fusion must be 'mean' or 'concat', got: {self.fusion}")

        self.head = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(fused_dim, self.pred_len),
        )

        self._last_gates = None

    def get_last_gates(self):
        return self._last_gates

    def export_all_expert_formulas(self):
        all_info = []
        for li, blk in enumerate(self.blocks):
            info = {"layer": li, "experts": blk.export_expert_formulas()}
            all_info.append(info)
        return all_info

    def _make_baseline(self, pv_hist: torch.Tensor):
        if self.baseline_mode == "last24_repeat":
            last24 = pv_hist[:, -24:, :]  # (B,24,1)
            reps = self.pred_len // 24
            rem = self.pred_len % 24
            base = last24.repeat(1, reps, 1)
            if rem > 0:
                base = torch.cat([base, last24[:, :rem, :]], dim=1)
            return base[:, :self.pred_len, :]

        last = pv_hist[:, -1:, :]
        return last.repeat(1, self.pred_len, 1)

    def forward(self, var_x: torch.Tensor, marker_x: torch.Tensor = None, return_gate: bool = False):
        B, L, F = var_x.shape
        assert L == self.hist_len, f"hist_len mismatch: got {L}, expected {self.hist_len}"
        assert F == self.var_num, f"var_num mismatch: got {F}, expected {self.var_num}"
        assert 0 <= self.pv_index < F, f"pv_index out of range: {self.pv_index}"

        pv = var_x[:, :, self.pv_index:self.pv_index + 1]  # (B,L,1)

        if self.use_pv_revin:
            pv_norm, mean, std = self.pv_revin.normalize(pv)
            x = var_x.clone()
            x[:, :, self.pv_index:self.pv_index + 1] = pv_norm
            pv_for_base = pv_norm
        else:
            x = var_x
            mean, std = None, None
            pv_for_base = pv

        baseline = self._make_baseline(pv_for_base)  # (B,T,1)

        tokens = self.token_encoder(x)  # (B,F,D)

        gates_all = []
        for blk in self.blocks:
            tokens, gates = blk(tokens)
            if return_gate:
                gates_all.append(gates.detach().cpu())

        if self.fusion == "mean":
            h = tokens.mean(dim=1)     # (B,D)
        else:
            h = tokens.reshape(B, -1)  # (B,F*D)

        delta = self.head(h).unsqueeze(-1)  # (B,T,1)
        y_norm = baseline + delta

        if self.use_pv_revin:
            y = self.pv_revin.denormalize(y_norm, mean=mean, std=std)
        else:
            y = y_norm

        if self.enforce_nonneg:
            y = torch.clamp(y, min=0.0)

        if return_gate:
            self._last_gates = gates_all
            return y, gates_all
        return y
