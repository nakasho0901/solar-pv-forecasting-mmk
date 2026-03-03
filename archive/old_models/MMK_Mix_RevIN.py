import torch
import torch.nn as nn
import torch.nn.functional as F
from easytsf.layer.kanlayer import KANInterfaceV2


class RevIN(nn.Module):
    r"""
    Reversible Instance Normalization (RevIN) の簡易実装

    - 入力窓（hist_len）の時間方向(dim=1)に沿って mean/std を計算
    - normalize → モデル → denormalize
    - affine=True の場合、正規化後に学習可能な scale/shift を適用

    形状:
      x: (B, L, N)
      mean/std: (B, 1, N)
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, self.num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, self.num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

    def normalize(self, x: torch.Tensor):
        """
        x: (B, L, N)
        return: x_norm, mean, std
        """
        mean = x.mean(dim=1, keepdim=True)  # (B, 1, N)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # (B, 1, N)
        std = torch.sqrt(var + self.eps)  # (B, 1, N)

        x_norm = (x - mean) / std

        if self.affine:
            x_norm = x_norm * self.gamma + self.beta

        return x_norm, mean, std

    def denormalize(self, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        """
        y: (B, T, N) 予測出力
        mean/std: (B, 1, N) normalize時の統計量
        """
        y_inv = y

        if self.affine:
            gamma = self.gamma.clamp(min=self.eps)
            y_inv = (y_inv - self.beta) / gamma

        y_inv = y_inv * std + mean
        return y_inv


class MoKLayer(nn.Module):
    """
    MoKLayer（Mixture of KAN）:
    - expertとして複数のKANを持ち、
    - gateネットワークで重み付けして合成する
    """
    def __init__(self, in_dim, out_dim, expert_config):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # expert_config 例: {"n_expert": 4, "kan_hp": {...}}
        self.n_expert = int(expert_config["n_expert"])
        kan_hp = expert_config["kan_hp"]

        self.experts = nn.ModuleList([
            KANInterfaceV2(in_dim, out_dim, **kan_hp) for _ in range(self.n_expert)
        ])

        # gate（入力から expert 重みを出す）
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, self.n_expert),
        )

    def forward(self, x):
        """
        x: (B, in_dim)
        return:
          y: (B, out_dim)
          scores: (B, n_expert) softmax済みゲート
        """
        gate_logits = self.gate(x)  # (B, n_expert)
        scores = torch.softmax(gate_logits, dim=-1)

        expert_outs = []
        for expert in self.experts:
            expert_outs.append(expert(x))  # (B, out_dim)

        # (B, n_expert, out_dim)
        stacked = torch.stack(expert_outs, dim=1)

        # scores: (B, n_expert) -> (B, n_expert, 1)
        weights = scores.unsqueeze(-1)

        y = (stacked * weights).sum(dim=1)  # (B, out_dim)
        return y, scores


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

        # PV予測ではRevINよりBatchNormの方がピークを維持しやすい
        self.bn = nn.BatchNorm1d(out_dim) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        y, scores = self.mok(x)  # scores: (B, n_expert)

        # 残差接続（次元が同じときだけ）
        if self.res_con:
            y = x + y

        # 正規化 + ドロップアウト
        y = self.dropout(self.bn(y))
        return y, scores


class MMK_Mix(nn.Module):
    """
    Multi-layer Mixture-of-KAN (MMK_Mix)

    入力:
      var_x: (B, L, N)
        B: batch
        L: hist_len
        N: 変数数（特徴量数）

    内部:
      変数ごとに独立に MoKBlock を通すために
      (B, L, N) -> (B*N, L) に変形して処理

    出力:
      prediction: (B, pred_len, N)

    ★追加機能（解釈性用）:
      - return_gate=True で forward すると各層のゲーティング重みを返す
      - get_last_gates() で直近 forward のゲートを取り出せる

    ★追加機能（RevIN）:
      - use_revin=True のとき、入力窓ごとの mean/std で正規化してから学習し、
        出力を逆正規化して返す（iTransformerの内部正規化に寄せる）
      - traingraph.py は変更不要で、CONFIGから use_revin 等を渡せる
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
        # --- RevIN 用（CONFIGで制御）---
        use_revin: bool = False,
        revin_eps: float = 1e-5,
        revin_affine: bool = True,
    ):
        super().__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.var_num = var_num
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.layer_hp = layer_hp

        # RevIN（必要なときだけ使う）
        self.use_revin = bool(use_revin)
        self.revin = RevIN(num_features=var_num, eps=revin_eps, affine=revin_affine)

        self.layers = nn.ModuleList()
        for i in range(layer_num):
            in_d = hist_len if i == 0 else hidden_dim
            out_d = pred_len if i == layer_num - 1 else hidden_dim
            self.layers.append(MoKBlock(in_d, out_d, layer_hp, use_norm))

        # 直近の forward で取得した gate を保存しておく領域（解釈性用）
        # 形式: List[Tensor]
        #   各要素は (B*N, n_expert) もしくは集約後 (B, N, n_expert)
        self._last_gates = None

    def get_last_gates(self):
        """
        直近の forward(return_gate=True) で保存されたゲートを返す。
        返り値:
          None または List[Tensor]
        """
        return self._last_gates

    def forward(self, var_x, marker_x=None, return_gate=False, gate_aggregate="var_mean"):
        """
        return_gate:
          False（デフォルト）: 従来通り prediction のみ返す（挙動は変えない）
          True : prediction に加えてゲートを返す

        gate_aggregate:
          "none"     : (B*N, n_expert) のまま返す（最も生データに近い）
          "var_mean" : 変数方向で平均して (B, n_expert) にする（時間×expertのヒートマップ向き）
          "var_keep" : (B, N, n_expert) に整形して返す（変数×expertを見たい時向き）
        """
        B, L, N = var_x.shape
        assert N == self.var_num, f"var_num mismatch: input N={N}, config var_num={self.var_num}"

        # ===== RevIN normalize（必要なときだけ）=====
        if self.use_revin:
            x_in, mean, std = self.revin.normalize(var_x)  # (B, L, N), (B,1,N), (B,1,N)
        else:
            x_in, mean, std = var_x, None, None

        # 入力次元を [Batch * Variable, Length] に変換
        x = x_in.transpose(1, 2).reshape(B * N, L)  # (B*N, hist_len)

        gates_per_layer = []  # 各層の gate を溜める（return_gate=True のときだけ）

        for layer in self.layers:
            x, gate_scores = layer(x)  # gate_scores: (B*N, n_expert)
            if return_gate:
                gates_per_layer.append(gate_scores.detach())

        # 出力を [Batch, Pred_len, Variable] に戻す
        prediction = x.reshape(B, N, -1).permute(0, 2, 1)  # (B, pred_len, N)

        # ===== RevIN denormalize（必要なときだけ）=====
        if self.use_revin:
            prediction = self.revin.denormalize(prediction, mean=mean, std=std)

        if not return_gate:
            return prediction

        # ----- gate の整形（解釈性用） -----
        processed = []
        for g in gates_per_layer:
            # g: (B*N, n_expert)
            if gate_aggregate == "none":
                processed.append(g.cpu())

            elif gate_aggregate == "var_keep":
                # (B*N, n_expert) -> (B, N, n_expert)
                processed.append(g.reshape(B, N, -1).cpu())

            elif gate_aggregate == "var_mean":
                # (B*N, n_expert) -> (B, N, n_expert) -> (B, n_expert)
                g_bn = g.reshape(B, N, -1)
                processed.append(g_bn.mean(dim=1).cpu())

            else:
                raise ValueError(f"Unknown gate_aggregate: {gate_aggregate}")

        # 直近ゲートとして保存（後で可視化スクリプトが取り出せる）
        self._last_gates = processed

        # prediction と gate（層ごとのリスト）を返す
        return prediction, processed
