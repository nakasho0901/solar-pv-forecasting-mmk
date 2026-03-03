# -*- coding: utf-8 -*-
"""
MMK_Mix_FusionPV.py

目的:
- 既存MMK_Mixは「変数ごとに独立」処理になり、多変量特徴がPVに効きにくい。
  （(B,L,N) -> (B*N,L) に潰しているため）  ※元実装の発想を保ったまま、入力融合に変更する

この実装:
- (B, L, N) を (B, L*N) に結合して MoKBlock を通し、
- PVのみを (B, pred_len, 1) で出力する。

オプション:
- enforce_nonneg=True で forward出力を ReLU（>=0）にする（推論用推奨）
- return_gate=True で層ごとのゲートも返す（解釈性）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from easytsf.layer.kanlayer import KANInterfaceV2


class MoKLayer(nn.Module):
    """
    Mixture-of-KAN Layer
    - gate: (B, in_features) -> (B, n_expert)
    - experts: 複数種類KAN
    - 出力は重み付き合成
    """
    def __init__(self, in_features: int, out_features: int, expert_config):
        super().__init__()
        self.n_expert = len(expert_config)

        self.gate = nn.Linear(in_features, self.n_expert)
        self.experts = nn.ModuleList([
            KANInterfaceV2(in_features, out_features, k[0], k[1]) for k in expert_config
        ])

    def forward(self, x: torch.Tensor):
        # x: (B, in_features)
        score = F.softmax(self.gate(x), dim=-1)  # (B, n_expert)
        expert_outputs = torch.stack([exp(x) for exp in self.experts], dim=-1)  # (B, out, n_expert)
        y = torch.einsum("BOE,BE->BO", expert_outputs, score)  # (B, out)
        return y, score


class MoKBlock(nn.Module):
    """
    MoKLayer + (optional) residual + BN + dropout
    """
    def __init__(self, in_dim: int, out_dim: int, expert_config, use_norm: bool = True):
        super().__init__()
        self.mok = MoKLayer(in_dim, out_dim, expert_config)
        self.res_con = (in_dim == out_dim)

        self.bn = nn.BatchNorm1d(out_dim) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor):
        y, scores = self.mok(x)
        if self.res_con:
            y = x + y
        y = self.dropout(self.bn(y))
        return y, scores


class MMK_Mix_FusionPV(nn.Module):
    """
    Multi-layer Mixture-of-KAN (Fusion -> PV only)

    Input:
      var_x: (B, L, N)

    Internal:
      flatten: (B, L*N)

    Output:
      prediction: (B, pred_len, 1)

    return_gate=True で gate を返す（層ごとの List[Tensor(B, n_expert)]）
    """
    def __init__(
        self,
        hist_len: int,
        pred_len: int,
        var_num: int,
        hidden_dim: int,
        layer_hp,
        layer_num: int,
        use_norm: bool = True,
        enforce_nonneg: bool = False,
    ):
        super().__init__()
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.var_num = int(var_num)
        self.hidden_dim = int(hidden_dim)
        self.layer_num = int(layer_num)
        self.layer_hp = layer_hp
        self.enforce_nonneg = bool(enforce_nonneg)

        in_dim0 = self.hist_len * self.var_num  # ★ 多変量融合
        self.layers = nn.ModuleList()

        for i in range(self.layer_num):
            in_d = in_dim0 if i == 0 else self.hidden_dim
            out_d = self.pred_len if i == self.layer_num - 1 else self.hidden_dim
            self.layers.append(MoKBlock(in_d, out_d, self.layer_hp, use_norm))

        self._last_gates = None

    def get_last_gates(self):
        return self._last_gates

    def forward(self, var_x: torch.Tensor, marker_x=None, return_gate: bool = False):
        # var_x: (B, L, N)
        B, L, N = var_x.shape
        assert L == self.hist_len, f"hist_len mismatch: input L={L}, config hist_len={self.hist_len}"
        assert N == self.var_num, f"var_num mismatch: input N={N}, config var_num={self.var_num}"

        # ★ 多変量融合: (B, L*N)
        x = var_x.reshape(B, L * N)

        gates_per_layer = []
        for layer in self.layers:
            x, gate_scores = layer(x)  # gate_scores: (B, n_expert)
            if return_gate:
                gates_per_layer.append(gate_scores.detach().cpu())

        # x: (B, pred_len) -> (B, pred_len, 1)
        pred = x.unsqueeze(-1)

        # 物理制約（推論用にON推奨。学習中はOFFでもOK）
        if self.enforce_nonneg:
            pred = F.relu(pred)

        if not return_gate:
            return pred

        self._last_gates = gates_per_layer
        return pred, gates_per_layer
