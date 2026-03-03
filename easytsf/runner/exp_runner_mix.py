# easytsf/runner/exp_runner_mix.py
import torch
import torch.nn as nn
from easytsf.runner.exp_runner import LTSFRunner

# --- 136kWのピークを仕留めるための重み付き損失関数 ---
class PeakWeightedMSE(nn.Module):
    def __init__(self, peak_threshold=1.0, weight=5.0):
        super().__init__()
        # 正規化された値で 1.0 (約53kW以上) をピークと判定
        self.peak_threshold = peak_threshold 
        self.weight = weight

    def forward(self, pred, target):
        loss = (pred - target) ** 2
        # 実測値（target）が高い箇所の重みを5倍に強化
        weights = torch.ones_like(target)
        weights[target > self.peak_threshold] = self.weight
        return (loss * weights).mean()

class LTSFRunnerMix(LTSFRunner):
    def configure_loss(self):
        # ピーク特化型損失関数を適用
        self.loss_function = PeakWeightedMSE(peak_threshold=1.0, weight=5.0)
        self.lb_weight = 0.1 # 負荷分散の重み

    def forward(self, batch, batch_idx):
        var_x, marker_x, var_y, marker_y = [_.float() for _ in batch]
        
        # 【物理的強化】日射量(Var 2)の入力を1.2倍強調してモデルに入力
        # 相関0.91の強力な日射量データを、モデルが「絶対に無視できない情報」として扱うようにします
        var_x[:, :, 2] = var_x[:, :, 2] * 1.2
        
        # モデルの推論（ここから先は従来通り）
        pred_output, _, _ = self.model(var_x, marker_x)
        return pred_output, var_y, marker_y

    def training_step(self, batch, batch_idx):
        pred, label, marker_y = self.forward(batch, batch_idx)
        loss_pred = self.loss_function(pred, label)
        
        # MMK_Mix独自の負荷分散損失(MoEのバランス)を加算
        loss_lb = self.model.get_load_balancing_loss()
        loss = loss_pred + self.lb_weight * loss_lb
        
        self.log("train/loss", loss, prog_bar=True)
        return loss