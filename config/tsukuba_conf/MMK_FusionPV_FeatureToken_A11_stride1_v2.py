# -*- coding: utf-8 -*-
# ============================================================
# MMK_FusionPV_FeatureToken (A11 stride1) v2 config - STABLE FIX
# ============================================================
# 目的：
# - val/maeが悪化(例: 19)した原因になりやすい要素を一旦弱めて安定化
# - 夜間をlossから完全除外しない（use_day_mask_in_loss=False）
# - GHI重み（proxy）を一旦OFF（use_ghi_loss_weight=False）
# - ピーク強調を弱める（peak_weight小、under_weightをほぼ対称に）
# - ピーク定義を上位5%に絞る（quantile=0.95）
# ============================================================

DATA_DIR = r"dataset_PV\prepared\96-96_fusionA11_96_96_stride1"
VAR_NUM = 12  # meta.json の feature_cols と一致
USE_4BASIS_LIST = True

exp_conf = {
    # -----------------------------
    # 基本
    # -----------------------------
    "model_name": "MMK_FusionPV_FeatureToken",
    "model_module": "easytsf.model.MMK_FusionPV_FeatureToken_v2",
    "model_class": "MMK_FusionPV_FeatureToken",

    # -----------------------------
    # データ
    # -----------------------------
    "data_dir": DATA_DIR,
    "hist_len": 96,
    "pred_len": 96,
    "var_num": VAR_NUM,

    "pv_index": 0,
    "ghi_index": 3,
    "ghi_boost": 1.0,

    # -----------------------------
    # モデル
    # -----------------------------
    "token_dim": 64,
    "layer_num": 3,
    "dropout": 0.05,
    "use_layernorm": True,
    "fusion": "mean",

    "baseline_mode": "last24_repeat",

    "use_pv_revin": True,
    "revin_eps": 1e-5,
    "enforce_nonneg": True,

    "layer_hp": (
        [
            ["KAN", 5],
            ["WavKAN", 5],
            ["TaylorKAN", 4],
            ["JacobiKAN", 4],
        ]
        if USE_4BASIS_LIST
        else {
            "n_expert": 4,
            "kan_hp": {"layer_type": "KAN", "hyperparam": 5},
        }
    ),

    # -----------------------------
    # 学習
    # -----------------------------
    "max_epochs": 30,
    "batch_size": 64,
    "num_workers": 0,

    "lr": 2e-5,
    "lr_step_size": 15,
    "lr_gamma": 0.5,

    "enable_early_stop": True,
    "early_stop_patience": 8,
    "early_stop_min_delta": 0.0,
    "gradient_clip_val": 1.0,
    "log_every_n_steps": 10,

    # 夜間0制約（runner側の marker_y を使う前提）
    # ※lossから夜を除外しない設定にしても、出力側を0に寄せる動きは維持される想定
    "force_night0": True,

    # -----------------------------
    # 損失（2段階）
    # -----------------------------
    "use_two_stage_loss": True,
    "warmup_epochs": 8,
    "huber_delta": 10.0,

    # =========================================================
    # ★ 安定化のための変更点（ここが重要）
    # =========================================================

    # --- Peakの定義（ピーク扱いを“上位5%”に絞る）
    "peak_mode": "quantile",
    "peak_threshold": 80.0,      # fixed の時だけ使う（単位: kWh）
    "peak_quantile": 0.95,       # 上位5%のみピーク扱い
    "peak_min_threshold": 20.0,  # 低すぎる閾値にならない安全弁（単位: kWh）

    # --- 重み（いったん弱くする）
    "peak_weight": 3.0,          # 8.0 → 2.0（強すぎて全体が壊れるのを防ぐ）
    "under_weight": 1.3,         # 2.0 → 1.2（不足だけ強罰を弱める）
    "over_weight": 1.1,          # そのまま（過大は通常罰）

    # --- 夜をlossから完全除外しない（重要）
    "use_day_mask_in_loss": False,

    # --- GHI重み（proxy）は一旦OFF（重要）
    "use_ghi_loss_weight": False,
    "ghi_loss_alpha": 0.0,
    "ghi_weight_index": 3,
    "ghi_clip_max": 1.0,
}
