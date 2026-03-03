# easytsf/model/iTransformer_peak.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from easytsf.layer.transformer import (
    Encoder, EncoderLayer, FullAttention, AttentionLayer, iTransformer_Embedder
)

class iTransformerPeak(nn.Module):
    def __init__(self, hist_len, pred_len, output_attention, d_model, dropout, factor, n_heads, d_ff, activation, e_layers, **kwargs):
        super().__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.output_attention = output_attention

        self.enc_embedding = iTransformer_Embedder(hist_len, d_model, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=output_attention),
                        d_model, n_heads
                    ),
                    d_model, d_ff, dropout=dropout, activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.projection = nn.Linear(d_model, self.pred_len)
        self.out_act = nn.Identity()

    def forecast(self, x_enc, x_mark_enc):
        means = x_enc.mean(1, keepdim=True)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = (x_enc - means) / stdev
        enc_out, attns = self.encoder(self.enc_embedding(x_enc, x_mark_enc), attn_mask=None)
        dec_out = self.projection(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        N = x_enc.size(-1)
        if dec_out.size(-1) > N:
            dec_out = dec_out[..., :N]
        dec_out = dec_out * stdev + means
        return dec_out, attns

    def forward(self, x_enc, x_mark_enc):
        """
        LTSFRunnerが期待する (予測値, アテンション, None) の3つを返します
        """
        out, attns = self.forecast(x_enc, x_mark_enc)
        pred = out[:, -self.pred_len:, :]
        return pred, attns, None