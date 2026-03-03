# easytsf/model/iTransformer.py
# -*- coding: utf-8 -*-
"""
iTransformer (pred_len固定対応版)
- hist_len に依存せず、pred_len ステップの出力を必ず返す。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from easytsf.layer.transformer import (
    Encoder, EncoderLayer, FullAttention, AttentionLayer, iTransformer_Embedder
)

class iTransformer(nn.Module):
    def __init__(
        self,
        hist_len,
        pred_len,
        output_attention,
        d_model,
        dropout,
        factor,
        n_heads,
        d_ff,
        activation,
        e_layers,
    ):
        super().__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.output_attention = output_attention

        # Embedding
        self.enc_embedding = iTransformer_Embedder(hist_len, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model,
                        n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Output projection (固定: d_model -> pred_len)
        self.projection = nn.Linear(d_model, self.pred_len)
        self.out_act = nn.Identity()

    def forecast(self, x_enc, x_mark_enc):
        """
        x_enc: [B, L, N]
        x_mark_enc: [B, L, ?]
        出力: [B, pred_len, N]
        """
        # --- 標準化 (batch 内でのスケーリング)
        means = x_enc.mean(1, keepdim=True)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = (x_enc - means) / stdev

        # --- エンコード ---
        enc_out, _ = self.encoder(self.enc_embedding(x_enc, x_mark_enc), attn_mask=None)  # [B, L, d_model]

        # --- プロジェクション ---
        dec_out = self.projection(enc_out)        # [B, L, pred_len]
        dec_out = dec_out.permute(0, 2, 1)        # [B, pred_len, L]
        N = x_enc.size(-1)
        if dec_out.size(-1) > N:
            dec_out = dec_out[..., :N]            # [B, pred_len, N]

        # --- 逆標準化 ---
        dec_out = dec_out * stdev + means
        return self.out_act(dec_out)

    def forward(self, x_enc, x_mark_enc):
        """
        Lightning 用 entrypoint
        """
        out = self.forecast(x_enc, x_mark_enc)
        # 最後の pred_len ステップに強制トリム（安全策）
        return out[:, -self.pred_len:, :]
