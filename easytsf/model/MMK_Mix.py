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
    MoKLayer（Mixture of KAN）

    この版は **2種類の layer_hp 形式**を受けます。

    (A) 4種類の基底関数を混ぜたい（あなたの1つ目/2つ目のconfig形式）
        expert_config = [
            ["KAN", 5],
            ["WavKAN", 5],
            ["TaylorKAN", 4],
            ["JacobiKAN", 4],
        ]
        - 各 expert を (layer_type, hyperparam) で指定
        - KANInterfaceV2(in_dim, out_dim, layer_type, hyperparam) を使って expert を作る

    (B) 従来の「同一タイプを n_expert 個」形式（互換用）
        expert_config = {"n_expert": 4, "kan_hp": {...}}
        - kan_hp の中に layer_type / hyperparam があればそれを使用
        - 無い場合は動作優先のフォールバック（ただし 4種類混合には(A)推奨）
    """
    def __init__(self, in_dim, out_dim, expert_config):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        # -----------------------------
        # (A) 4種類混合: list[[layer_type, hyperparam], ...]
        # -----------------------------
        if isinstance(expert_config, (list, tuple)):
            if len(expert_config) == 0:
                raise ValueError("[ERROR] layer_hp(list) が空です。")
            self.n_expert = len(expert_config)

            experts = []
            for i, spec in enumerate(expert_config):
                if not (isinstance(spec, (list, tuple)) and len(spec) == 2):
                    raise TypeError(f"[ERROR] layer_hp[{i}] は [layer_type, hyperparam] を想定: {spec}")
                layer_type = str(spec[0])
                hyperparam = spec[1]
                experts.append(KANInterfaceV2(self.in_dim, self.out_dim, layer_type, hyperparam))

            self.experts = nn.ModuleList(experts)

        # -----------------------------
        # (B) 従来形式: dict {"n_expert":..., "kan_hp": {...}}
        # -----------------------------
        elif isinstance(expert_config, dict):
            if ("n_expert" not in expert_config) or ("kan_hp" not in expert_config):
                raise KeyError("[ERROR] layer_hp(dict) には 'n_expert' と 'kan_hp' が必要です。")

            self.n_expert = int(expert_config["n_expert"])
            kan_hp = dict(expert_config["kan_hp"])  # コピー

            # まず layer_type / hyperparam が入っていればそれを優先
            layer_type = kan_hp.get("layer_type", None)
            hyperparam = kan_hp.get("hyperparam", None)

            # 無ければフォールバック（古い dict config を“とりあえず動かす”）
            if layer_type is None:
                layer_type = "KAN"
            if hyperparam is None:
                if "grid_size" in kan_hp:
                    hyperparam = kan_hp["grid_size"]
                elif "degree" in kan_hp:
                    layer_type = "JacobiKAN"
                    hyperparam = kan_hp["degree"]
                elif "order" in kan_hp:
                    layer_type = "TaylorKAN"
                    hyperparam = kan_hp["order"]
                else:
                    hyperparam = 5

            self.experts = nn.ModuleList([
                KANInterfaceV2(self.in_dim, self.out_dim, str(layer_type), hyperparam)
                for _ in range(self.n_expert)
            ])

        else:
            raise TypeError(f"[ERROR] layer_hp は list か dict を想定: {type(expert_config)}")

        # -----------------------------
        # gate（入力から expert 重みを出す）
        # -----------------------------
        self.gate = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.ReLU(),
            nn.Linear(self.in_dim, self.n_expert),
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

        expert_outs = [expert(x) for expert in self.experts]  # list of (B, out_dim)
        stacked = torch.stack(expert_outs, dim=1)             # (B, n_expert, out_dim)

        weights = scores.unsqueeze(-1)                        # (B, n_expert, 1)
        y = (stacked * weights).sum(dim=1)                    # (B, out_dim)
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
        出力を逆正規化して返す
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

        # -----------------------------
        # 変数ごとに同じ MLP(MoKBlock) を当てるのではなく、
        # (B*N, L) として「変数をバッチに畳み込んで」処理する設計
        # -----------------------------
        self.layers = nn.ModuleList()
        in_dim = hist_len
        for _ in range(layer_num):
            self.layers.append(MoKBlock(in_dim, hidden_dim, layer_hp, use_norm=use_norm))
            in_dim = hidden_dim

        # 最終: hidden_dim -> pred_len
        self.proj = nn.Linear(hidden_dim, pred_len)

        # 解釈性用（直近forwardのgateを保持）
        self._last_gates = None

    def get_last_gates(self):
        """
        直近 forward で保存された gate を返す。
        形状: list[length=layer_num] of (B*N, n_expert)
        """
        return self._last_gates

    def forward(self, var_x, marker_x=None, return_gate=False):
        """
        var_x: (B, L, N)
        return:
          pred: (B, pred_len, N)
          (option) gates: list of (B*N, n_expert)
        """
        B, L, N = var_x.shape
        assert L == self.hist_len, f"hist_len mismatch: got {L}, expected {self.hist_len}"
        assert N == self.var_num, f"var_num mismatch: got {N}, expected {self.var_num}"

        # RevIN: 入力窓で正規化（モデルに入れる前）
        if self.use_revin:
            x_norm, mean, std = self.revin.normalize(var_x)
        else:
            x_norm = var_x
            mean, std = None, None

        # (B, L, N) -> (B*N, L)
        x = x_norm.permute(0, 2, 1).contiguous().view(B * N, L)

        gates_all = []
        for layer in self.layers:
            x, scores = layer(x)  # x: (B*N, hidden_dim), scores: (B*N, n_expert)
            gates_all.append(scores)

        # (B*N, hidden_dim) -> (B*N, pred_len)
        y = self.proj(x)

        # (B*N, pred_len) -> (B, N, pred_len) -> (B, pred_len, N)
        y = y.view(B, N, self.pred_len).permute(0, 2, 1).contiguous()

        # RevIN: 逆正規化（出力に適用）
        if self.use_revin:
            y = self.revin.denormalize(y, mean=mean, std=std)

        # 直近 gate を保存
        self._last_gates = gates_all

        if return_gate:
            return y, gates_all
        return y
