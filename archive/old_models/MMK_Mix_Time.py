import torch
import torch.nn as nn
import torch.nn.functional as F

from easytsf.layer.kanlayer import KANInterfaceV2


class MoKBlock(nn.Module):
    """
    MoKLayer（複数KAN expert + gate）を 1ブロックとしてまとめたもの。
    """
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
    """
    Mixture-of-KAN Layer
    """
    def __init__(self, in_features, out_features, expert_config):
        super().__init__()
        self.n_expert = len(expert_config)
        self.gate = nn.Linear(in_features, self.n_expert)
        self.experts = nn.ModuleList([
            KANInterfaceV2(in_features, out_features, k[0], k[1]) for k in expert_config
        ])

    def forward(self, x):
        score = F.softmax(self.gate(x), dim=-1)  # (B, n_expert)
        expert_outputs = torch.stack([exp(x) for exp in self.experts], dim=-1)  # (B, out_features, n_expert)
        y = torch.einsum("BOE,BE->BO", expert_outputs, score)  # (B, out_features)
        return y, score


class MMK_Mix_Time(nn.Module):
    """
    MMK_Mix の派生版：marker_x（時間特徴）を実際に使う

    入力:
      var_x   : (B, L, N)
      marker_x: (B, L, M)  ※tod/dow等（use_time_feature=True の時に渡される想定）

    仕組み:
      marker_x を (B, L*M) に flatten し、LazyLinear で (B, L) に圧縮。
      それを (B, N, L) に複製して (B*N, L) へ変形し、var系列入力に加算する。
      → 「今が何時か」をモデルに必ず渡せるので、日の出/日没・昼ピークの形が学習しやすくなる。

    注意:
      - 既存 MMK_Mix を壊さないため、別クラス・別ファイルに分離
      - time_scale を入れて、時間埋め込みが強すぎる場合に抑えられる
    """
    def __init__(
        self,
        hist_len,
        pred_len,
        var_num,
        hidden_dim,
        layer_hp,
        layer_num,
        use_norm=True,
        use_time_feature=True,
        time_scale=0.1,
    ):
        super().__init__()
        self.hist_len = int(hist_len)
        self.pred_len = int(pred_len)
        self.var_num = int(var_num)
        self.hidden_dim = int(hidden_dim)
        self.layer_num = int(layer_num)
        self.layer_hp = layer_hp

        self.use_time_feature = bool(use_time_feature)
        self.time_scale = float(time_scale)

        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            in_d = self.hist_len if i == 0 else self.hidden_dim
            out_d = self.pred_len if i == self.layer_num - 1 else self.hidden_dim
            self.layers.append(MoKBlock(in_d, out_d, self.layer_hp, use_norm))

        # marker_x: (B, L, M) -> flatten (B, L*M) -> (B, L)
        # M はconfig次第で変動するので LazyLinear で受ける
        if self.use_time_feature:
            self.time_proj = nn.LazyLinear(self.hist_len)

        self._last_gates = None

    def get_last_gates(self):
        return self._last_gates

    def forward(self, var_x, marker_x=None, return_gate=False, gate_aggregate="var_mean"):
        B, L, N = var_x.shape
        assert N == self.var_num, f"var_num mismatch: input N={N}, config var_num={self.var_num}"
        assert L == self.hist_len, f"hist_len mismatch: input L={L}, config hist_len={self.hist_len}"

        x = var_x.transpose(1, 2).reshape(B * N, L)  # (B*N, L)

        # ===== 時間特徴を加算 =====
        if self.use_time_feature and (marker_x is not None):
            mk = marker_x.reshape(B, -1)                 # (B, L*M)
            time_emb = self.time_proj(mk)                # (B, L)
            time_emb = self.time_scale * time_emb        # スケールで効きすぎを防ぐ

            time_add = time_emb.unsqueeze(1).repeat(1, N, 1).reshape(B * N, L)  # (B*N, L)
            x = x + time_add

        gates_per_layer = []

        for layer in self.layers:
            x, gate_scores = layer(x)  # (B*N, dim)
            if return_gate:
                gates_per_layer.append(gate_scores.detach())

        prediction = x.reshape(B, N, -1).permute(0, 2, 1)  # (B, pred_len, N)

        if not return_gate:
            return prediction

        processed = []
        for g in gates_per_layer:
            if gate_aggregate == "none":
                processed.append(g.cpu())
            elif gate_aggregate == "var_keep":
                processed.append(g.reshape(B, N, -1).cpu())
            elif gate_aggregate == "var_mean":
                g_bn = g.reshape(B, N, -1)
                processed.append(g_bn.mean(dim=1).cpu())
            else:
                raise ValueError(f"Unknown gate_aggregate: {gate_aggregate}")

        self._last_gates = processed
        return prediction, processed
