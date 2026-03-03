import torch
import torch.nn as nn
import torch.nn.functional as F
from easytsf.layer.kanlayer import KANInterfaceV2


# ============================================================
# RevIN (Reversible Instance Normalization)
#  - sample-wise, variable-wise normalization over time axis
#  - reversible (denorm restores original scale)
# ============================================================
class RevIN(nn.Module):
    """
    x: (B, L, N)

    - norm:   x -> (x - mu) / sigma
    - denorm: y -> y * sigma + mu

    Notes:
    - mu, sigma are computed from the INPUT window (hist_len) only.
    - sigma is clamped by sigma_min to avoid blow-up in near-constant windows
      (common for PV at night).
    """
    def __init__(self, eps: float = 1e-5, sigma_min: float = 1e-3):
        super().__init__()
        self.eps = eps
        self.sigma_min = sigma_min
        self._mu = None
        self._sigma = None

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        if mode == "norm":
            # mean/std over time dimension (L)
            mu = x.mean(dim=1, keepdim=True)  # (B, 1, N)
            var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, N)
            sigma = torch.sqrt(var + self.eps)
            sigma = torch.clamp(sigma, min=self.sigma_min)

            # store for denorm
            self._mu = mu
            self._sigma = sigma

            return (x - mu) / sigma

        elif mode == "denorm":
            assert self._mu is not None and self._sigma is not None, \
                "RevIN denorm called before norm. Call mode='norm' first."
            return x * self._sigma + self._mu

        else:
            raise ValueError(f"Unknown RevIN mode: {mode}")


class MoKBlock(nn.Module):
    """
    MoKLayer（複数KAN expert + gate）を 1ブロックとしてまとめたもの。

    目的:
    - 残差接続（in_dim == out_dimのとき）
    - BatchNorm（PV予測ではピーク維持に寄与しやすい）
    - Dropout

    forwardの戻り値:
    - y: (B, out_dim)
    - scores: (B, n_expert)  ※softmax済みゲーティング重み
    """
    def __init__(self, in_dim, out_dim, expert_config, use_norm=True):
        super().__init__()
        self.mok = MoKLayer(in_dim, out_dim, expert_config)
        self.res_con = (in_dim == out_dim)

        # 現行MMK_Mixと同じ（正規化以外は構造を変えない）
        self.bn = nn.BatchNorm1d(out_dim) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        y, scores = self.mok(x)  # scores: (B, n_expert)

        # 残差接続（次元が同じときだけ）
        if self.res_con:
            y = x + y

        # 正規化 + ドロップアウト（現行と同じ）
        y = self.dropout(self.bn(y))
        return y, scores


class MoKLayer(nn.Module):
    """
    Mixture-of-KAN Layer

    - gate: 入力から expert の重み（確率）を算出
    - experts: 複数種類の KAN（Spline / Wavelet / Taylor / Jacobi など）
    - 出力は expert の重み付き合成

    forwardの戻り値:
    - y: (B, out_features)
    - score: (B, n_expert)  ※softmax済みゲーティング重み
    """
    def __init__(self, in_features, out_features, expert_config):
        super().__init__()
        self.n_expert = len(expert_config)

        # gate: (B, in_features) -> (B, n_expert)
        self.gate = nn.Linear(in_features, self.n_expert)

        # experts: 各 expert は KANInterfaceV2(in_features -> out_features)
        self.experts = nn.ModuleList([
            KANInterfaceV2(in_features, out_features, k[0], k[1]) for k in expert_config
        ])

    def forward(self, x):
        """
        x: (B, in_features)
        """
        # gate score（確率）
        score = F.softmax(self.gate(x), dim=-1)  # (B, n_expert)

        # 各 expert 出力をstack
        # exp(x): (B, out_features)
        # -> stack: (B, out_features, n_expert)
        expert_outputs = torch.stack([exp(x) for exp in self.experts], dim=-1)

        # 重み付き合成
        # (B, out_features, n_expert) と (B, n_expert) を合成 -> (B, out_features)
        y = torch.einsum("BOE,BE->BO", expert_outputs, score)
        return y, score


class MMK_Mix_RevIN_bn(nn.Module):
    """
    MMK_Mix_RevIN_bn

    - 構造は現行MMK_Mixと同一（MoKBlock: residual + BatchNorm + Dropout）
    - 変更点は「RevINを入力前後に挟む」ことだけ

    入力:
      var_x: (B, L, N)
        B: batch
        L: hist_len
        N: 変数数（特徴量数）

    出力:
      prediction: (B, pred_len, N)

    ★追加機能（解釈性用）:
      - return_gate=True で forward すると各層のゲーティング重みを返す
      - get_last_gates() で直近 forward のゲートを取り出せる
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
        revin_eps=1e-5,
        revin_sigma_min=1e-3,
    ):
        super().__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.var_num = var_num
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.layer_hp = layer_hp

        # RevIN（追加点）
        self.revin = RevIN(eps=revin_eps, sigma_min=revin_sigma_min)

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
        assert N == self.var_num, f"var_num mismatch: input N={N}, config var_num={self.var_num}"

        # ---------------------------
        # RevIN+（入力窓に基づく正規化）
        # ---------------------------
        x = self.revin(var_x, mode="norm")  # (B, L, N)

        # 入力次元を [Batch * Variable, Length] に変換（現行と同じ）
        x = x.transpose(1, 2).reshape(B * N, L)  # (B*N, hist_len)

        gates_per_layer = []

        for layer in self.layers:
            x, gate_scores = layer(x)  # gate_scores: (B*N, n_expert)
            if return_gate:
                gates_per_layer.append(gate_scores.detach())

        # 出力を [Batch, Pred_len, Variable] に戻す（現行と同じ）
        prediction = x.reshape(B, N, -1).permute(0, 2, 1)  # (B, pred_len, N)

        # ---------------------------
        # RevIN-（元スケールへ戻す）
        # ---------------------------
        prediction = self.revin(prediction, mode="denorm")  # (B, pred_len, N)

        if not return_gate:
            return prediction

        # ----- gate の整形（解釈性用） -----
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
