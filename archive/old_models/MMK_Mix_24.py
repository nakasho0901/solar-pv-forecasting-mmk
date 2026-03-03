import torch
import torch.nn as nn
import torch.nn.functional as F
from easytsf.layer.kanlayer import KANInterfaceV2


class MoKBlock(nn.Module):
    def __init__(self, in_dim, out_dim, expert_config, use_norm=True):
        super().__init__()
        self.mok = MoKLayer(in_dim, out_dim, expert_config)
        self.res_con = (in_dim == out_dim)
        self.bn = nn.BatchNorm1d(out_dim) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        y, scores = self.mok(x)
        if self.res_con:
            y = x + y
        y = self.dropout(self.bn(y))
        return y, scores


class MoKLayer(nn.Module):
    def __init__(self, in_features, out_features, expert_config):
        super().__init__()
        self.n_expert = len(expert_config)
        self.gate = nn.Linear(in_features, self.n_expert)
        self.experts = nn.ModuleList([
            KANInterfaceV2(in_features, out_features, k[0], k[1])
            for k in expert_config
        ])

    def forward(self, x):
        score = F.softmax(self.gate(x), dim=-1)
        expert_outputs = torch.stack([exp(x) for exp in self.experts], dim=-1)
        y = torch.einsum("BOE,BE->BO", expert_outputs, score)
        return y, score


class MMK_Mix_24(nn.Module):
    """
    MMK_Mix (24-24, PV-in)
    Input : (B, hist_len, var_num)
    Output: (B, pred_len, var_num)
    """
    def __init__(self, hist_len, pred_len, var_num,
                 hidden_dim, layer_hp, layer_num, use_norm=True):
        super().__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.var_num = var_num

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            in_d = hist_len if i == 0 else hidden_dim
            out_d = pred_len if i == layer_num - 1 else hidden_dim
            self.layers.append(MoKBlock(in_d, out_d, layer_hp, use_norm))

        self._last_gates = None

    def get_last_gates(self):
        return self._last_gates

    def forward(self, var_x, marker_x=None, return_gate=False, gate_aggregate="var_mean"):
        B, L, N = var_x.shape
        assert N == self.var_num, f"var_num mismatch: {N} vs {self.var_num}"

        x = var_x.transpose(1, 2).reshape(B * N, L)

        gates = []
        for layer in self.layers:
            x, g = layer(x)
            if return_gate:
                gates.append(g.detach())

        pred = x.reshape(B, N, -1).permute(0, 2, 1)

        if not return_gate:
            return pred

        processed = []
        for g in gates:
            g_bn = g.reshape(B, N, -1)
            processed.append(g_bn.mean(dim=1).cpu())

        self._last_gates = processed
        return pred, processed
