# ============================================================
# MMK_FusionPV_Flatten（FusionPV構造 + 残差 + PV RevIN）
# ============================================================

# ---- ここだけ切り替えやすく ----
DATA_DIR = r"dataset_PV\prepared\96-96_fusionA11_96_96_stride1"  # ←stride1/12特徴ならこれ
VAR_NUM = 12  # ← train_x が (..., 96, 12) なので必須

# A11 features（あなたの meta.json 出力）
# ['pv_kwh','temp_c','rh_pct','ghi_wm2','hour_sin','hour_cos','is_daylight',
#  'lag_pv_kwh_24h','lag_ghi_wm2_1h','lag_ghi_wm2_2h','lag_ghi_wm2_3h','lag_ghi_wm2_24h']

USE_4BASIS_LIST = True

exp_conf = {
    # -----------------------------
    # 基本
    # -----------------------------
    "model_name": "MMK_FusionPV_FeatureToken",

    # -----------------------------
    # データ
    # -----------------------------
    "data_dir": DATA_DIR,
    "hist_len": 96,
    "pred_len": 96,
    "var_num": VAR_NUM,

    "pv_index": 0,
    "ghi_index": 3,
    "ghi_boost": 1.0,  # 入力側で GHI を単純に強調したいなら >1 も可（まずは 1.0）

    # -----------------------------
    # モデル
    # -----------------------------
    "hidden_dim": 128,
    "layer_num": 3,
    "dropout": 0.1,
    "use_layernorm": True,

    # 残差ベースライン（モデル側仕様に合わせる）
    "baseline_mode": "last24_repeat",

    # PV RevIN
    "use_pv_revin": True,
    "revin_eps": 1e-5,

    # 出力制約（モデル側）
    "enforce_nonneg": True,

    # 4基底（KAN/WavKAN/TaylorKAN/JacobiKAN）
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

    "lr": 3e-5,
    "lr_step_size": 15,
    "lr_gamma": 0.5,

    "enable_early_stop": True,
    "early_stop_patience": 8,
    "early_stop_min_delta": 0.0,
    "gradient_clip_val": 1.0,
    "log_every_n_steps": 10,

    # 夜間0（runner側で実施：モデル側にも同等制約を入れてるなら二重に注意）
    "force_night0": True,

    # -----------------------------
    # 損失：2段階（前半Huber→後半PeakWeighted）
    # -----------------------------
    "use_two_stage_loss": True,
    "warmup_epochs": 5,
    "huber_delta": 10.0,
    "peak_threshold": 1.0,
    "peak_weight": 10.0,

    # -----------------------------
    # ★追加：GHI重み付き損失（PeakWeightedMSE の中で有効化される）
    # -----------------------------
    "use_ghi_loss_weight": True,
    "ghi_loss_alpha": 3.0,      # 2〜5 目安
    "ghi_weight_index": 3,      # 通常は ghi_index と同じ（ghi_wm2 が入ってる列）
    "ghi_clip_max": 1.0,
}
