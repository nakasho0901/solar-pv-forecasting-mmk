# config/tsukuba_conf/MMK_Mix_RevIN_PV_96.py
# 方針B：noscale + RevIN + is_daylightはXから除外

exp_conf = {
  # =========================
  # モデル・データ
  # =========================
  "model_name": "MMK_Mix_RevIN_bn",
  "dataset_name": "Tsukuba",

  # ★ noscale + noday データセット
  "data_dir": "dataset_PV/prepared/96-96_pvin_noscale_noday",

  "train_x": "train_x.npy", "train_y": "train_y.npy",
  "val_x": "val_x.npy",     "val_y": "val_y.npy",
  "test_x": "test_x.npy",   "test_y": "test_y.npy",

  # marker_y は昼夜マスクとして使用
  "train_marker_y": "train_marker_y.npy",
  "val_marker_y":   "val_marker_y.npy",
  "test_marker_y":  "test_marker_y.npy",

  "ghi_threshold": 1.0,

  # =========================
  # 時系列設定
  # =========================
  "hist_len": 96,
  "fut_len": 96,
  "pred_len": 96,

  # ★ is_daylight を X から落とした後の特徴数に合わせる
  # meta.json の feature_cols の長さと必ず一致させること
  "var_num": 7,

  # =========================
  # MMK / MoK 設定
  # =========================
  "hidden_dim": 96,
  "layer_num": 3,
  "layer_type": "MoK",

  # ★ 4基底（論文MMKと同一思想）
  "layer_hp": [
      ["KAN", 5],
      ["WavKAN", 5],
      ["TaylorKAN", 4],
      ["JacobiKAN", 4],
  ],

  # =========================
  # 時刻特徴
  # =========================
  "use_time_feature": True,
  "norm_time_feature": True,
  "time_feature_cls": ["tod", "dow"],

  # =========================
  # 学習設定
  # =========================
  "batch_size": 64,
  "lr": 1e-4,
  "optimizer_weight_decay": 1e-4,

  "max_epochs": 30,
  "val_metric": "val/loss",

  "enable_early_stop": True,
  "early_stop_patience": 8,
  "early_stop_min_delta": 0.0001,

  "gradient_clip_val": 1.0,

  "lr_step_size": 15,
  "lr_gamma": 0.5,

  "ckpt_save_top_k": 5,
  "ckpt_save_last": True,

  # =========================
  # PV向け制約
  # =========================
  "enforce_nonneg": True,
  "force_night0": True,
  "use_marker_mask": True,

  # ★ meta.json と一致させる（必須）
"var_num": 22,

# ★ traingraph_revin.py が使う index（推奨：明示）
"pv_index": 0,
"ghi_index": 3,

# （任意）GHIブースト倍率を変えたいなら
# "ghi_boost": 1.3,

}
