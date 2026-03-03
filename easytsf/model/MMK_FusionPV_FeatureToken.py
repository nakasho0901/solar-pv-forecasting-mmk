# -*- coding: utf-8 -*-
"""
MMK_FusionPV_FeatureToken.py

解釈性重視 FusionPV モデル（FeatureToken + Feature-wise Gating）
- 入力: var_x (B, L, F) を「特徴量ごと」に時間方向へ要約して token 化: (B, F, D)
- ゲート: 各特徴量tokenごとに expert 重みを出す: (B, F, E)
- expert: KANInterfaceV2 を expertごとに layer_type を変えて混合可能
- 融合: token を集約して PV のみを予測 (B, pred_len, 1)
- 残差予測: baseline (last24 repeat) + delta
- PV RevIN: PV列だけ normalize -> baseline/delta をその空間で -> denorm
- 出力制約: clamp(min=0)
- heatmap用: forward(return_gate=True) で gate を返す

注意:
- KANInterfaceV2 の詳細（式抽出API等）は kanlayer.py 次第。
  ここでは「式抽出フック」を用意し、存在するメソッドを呼ぶ形にしている。
"""

import torch
import torch.nn as nn

from easytsf.layer.kanlayer import KANInterfaceV2


# ============================================================
# PV専用 RevIN
# ============================================================
class PVRevIN(nn.Module):
    """PV専用 RevIN（B, L, 1 を時間方向に正規化）"""
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)

    def normalize(self, pv: torch.Tensor):
        # pv: (B, L, 1)
        mean = pv.mean(dim=1, keepdim=True)  # (B, 1, 1)
        var = pv.var(dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)     # (B, 1, 1)
        pv_norm = (pv - mean) / std
        return pv_norm, mean, std

    def denormalize(self, y_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        return y_norm * std + mean


# ============================================================
# FeatureToken Encoder（特徴量ごとに時間方向を要約して token 化）
# ============================================================
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

        # 各特徴量に同じエンコーダを共有（解釈性とパラメータ節約）
        self.encoder = nn.Sequential(
            nn.Linear(self.hist_len, self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.token_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        B, L, F = x.shape
        assert L == self.hist_len, f"hist_len mismatch: got {L}, expected {self.hist_len}"
        assert F == self.var_num, f"var_num mismatch: got {F}, expected {self.var_num}"

        # (B, L, F) -> (B, F, L)
        xf = x.permute(0, 2, 1).contiguous()
        # (B*F, L)
        xf = xf.view(B * F, L)
        # (B*F, D)
        tf = self.encoder(xf)
        # (B, F, D)
        tf = tf.view(B, F, self.token_dim)
        return tf


# ============================================================
# Expert MoE for each feature token
# ============================================================
class FeatureMoK(nn.Module):
    """
    特徴量tokenごとにゲートを出し、expertを混合して token を更新する。

    入力:  tokens (B, F, D)
    出力:  tokens' (B, F, D), gates (B, F, E)

    layer_hp:
      - list形式: [["KAN",5],["WavKAN",5],["TaylorKAN",4],["JacobiKAN",4]]
      - dict形式: {"n_expert":4, "kan_hp":{"layer_type":"KAN","hyperparam":5}}
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

        # 特徴量tokenごとにゲート (D -> E)
        self.gate = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.n_expert),
        )

    def forward(self, tokens: torch.Tensor):
        # tokens: (B, F, D)
        B, F, D = tokens.shape
        assert D == self.dim

        # (B*F, D)
        t = tokens.reshape(B * F, D)

        # gate: (B*F, E) -> (B, F, E)
        logits = self.gate(t)
        scores = torch.softmax(logits, dim=-1).view(B, F, self.n_expert)

        # experts: 各expertで (B*F, D)
        outs = [ex(t) for ex in self.experts]          # list[(B*F, D)]
        stacked = torch.stack(outs, dim=1)             # (B*F, E, D)
        # (B*F, E, 1)
        w = scores.view(B * F, self.n_expert).unsqueeze(-1)
        # mixture: (B*F, D)
        y = (stacked * w).sum(dim=1)
        # (B, F, D)
        y = y.view(B, F, D)

        return y, scores

    # --- 式抽出フック（KANInterfaceV2が対応している場合に呼べる） ---
    def export_expert_formulas(self):
        """
        expertごとの式/パラメータを抽出（KANInterfaceV2側のAPIに依存）
        戻り値は dict のリスト（expert単位）にする。
        """
        formulas = []
        for i, ex in enumerate(self.experts):
            info = {"expert_id": i, "class": ex.__class__.__name__}

            # よくある候補メソッドを試す（存在しなければスキップ）
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

            # 何も無ければ parameters 名だけでも残す（後で実装を見て抽出器を書くため）
            if "formula" not in info:
                info["param_names"] = [n for (n, _) in ex.named_parameters()]

            formulas.append(info)
        return formulas


class FeatureMoKBlock(nn.Module):
    """FeatureMoK + 残差 + 正規化"""
    def __init__(self, dim: int, layer_hp, use_norm: bool = True, dropout: float = 0.1):
        super().__init__()
        self.mok = FeatureMoK(dim, layer_hp)
        self.norm = nn.LayerNorm(dim) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor):
        # tokens: (B, F, D)
        y, gates = self.mok(tokens)          # y: (B,F,D), gates:(B,F,E)
        tokens = tokens + y
        tokens = self.dropout(self.norm(tokens))
        return tokens, gates

    def export_expert_formulas(self):
        return self.mok.export_expert_formulas()


# ============================================================
# 本体モデル
# ============================================================
class MMK_FusionPV_FeatureToken(nn.Module):
    """
    FusionPV（特徴量別ゲート版）

    入力:
      var_x: (B, hist_len, var_num)
      marker_x: (B, hist_len, marker_dim)  ※未使用（特徴量にhour_sin/cos等を含める方針）
    出力:
      pred: (B, pred_len, 1)

    返り値:
      return_gate=True のとき (pred, gates)
        gates: list[Tensor]  各層の gate (B, F, E)
    """
    def __init__(
        self,
        hist_len: int,
        pred_len: int,
        var_num: int,
        pv_index: int = 0,

        # token / model dim
        token_dim: int = 64,
        layer_num: int = 3,
        layer_hp=None,
        use_layernorm: bool = True,
        dropout: float = 0.1,

        # fusion strategy
        fusion: str = "mean",  # "mean" or "concat"

        # residual baseline
        baseline_mode: str = "last24_repeat",

        # PV RevIN
        use_pv_revin: bool = True,
        revin_eps: float = 1e-5,

        # output constraint
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

        # 1) token encoder（特徴量ごと）
        self.token_encoder = FeatureTokenEncoder(
            hist_len=self.hist_len,
            var_num=self.var_num,
            token_dim=self.token_dim,
            dropout=self.dropout,
        )

        # 2) MoK blocks（特徴量ごとゲート）
        self.blocks = nn.ModuleList([
            FeatureMoKBlock(self.token_dim, self.layer_hp, use_norm=self.use_layernorm, dropout=self.dropout)
            for _ in range(self.layer_num)
        ])

        # 3) fusion head
        if self.fusion == "mean":
            fused_dim = self.token_dim
            self.fuser = None
        elif self.fusion == "concat":
            fused_dim = self.var_num * self.token_dim
            self.fuser = None
        else:
            raise ValueError(f"[ERROR] fusion must be 'mean' or 'concat', got: {self.fusion}")

        # 4) pred head（deltaを出す）
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
        """
        各層の expert の式/パラメータ情報を抽出（可能な範囲）
        """
        all_info = []
        for li, blk in enumerate(self.blocks):
            info = {"layer": li, "experts": blk.export_expert_formulas()}
            all_info.append(info)
        return all_info

    def _make_baseline(self, pv_hist: torch.Tensor):
        # pv_hist: (B, L, 1)
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

        # PV列
        pv = var_x[:, :, self.pv_index:self.pv_index + 1]  # (B,L,1)

        # PV RevIN（PVだけ正規化して、var_x 内PV列に戻す）
        if self.use_pv_revin:
            pv_norm, mean, std = self.pv_revin.normalize(pv)
            x = var_x.clone()
            x[:, :, self.pv_index:self.pv_index + 1] = pv_norm
            pv_for_base = pv_norm
        else:
            x = var_x
            mean, std = None, None
            pv_for_base = pv

        # baseline（正規化空間）
        baseline = self._make_baseline(pv_for_base)  # (B,T,1)

        # 1) tokens: (B,F,D)
        tokens = self.token_encoder(x)

        # 2) MoK blocks
        gates_all = []
        for blk in self.blocks:
            tokens, gates = blk(tokens)  # gates: (B,F,E)
            if return_gate:
                gates_all.append(gates.detach().cpu())

        # 3) fuse tokens -> (B, fused_dim)
        if self.fusion == "mean":
            h = tokens.mean(dim=1)  # (B,D)
        else:  # concat
            h = tokens.reshape(B, -1)  # (B, F*D)

        # 4) delta: (B,T,1)
        delta = self.head(h).unsqueeze(-1)

        # 5) predict in norm-space
        y_norm = baseline + delta

        # 6) denorm
        if self.use_pv_revin:
            y = self.pv_revin.denormalize(y_norm, mean=mean, std=std)
        else:
            y = y_norm

        # 7) nonneg constraint (night0 は runner で marker_y を掛ける)
        if self.enforce_nonneg:
            y = torch.clamp(y, min=0.0)

        if return_gate:
            self._last_gates = gates_all
            return y, gates_all
        return y
