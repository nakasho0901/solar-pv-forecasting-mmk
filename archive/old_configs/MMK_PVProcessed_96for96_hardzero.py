# -*- coding: utf-8 -*-
# MMK hardzero 96->96 (dataset_PV/processed_hardzero)
exp_conf = {
  "model_name": "MMK",
  "dataset_name": "Tsukuba",

  "data_dir": "dataset_PV/processed_hardzero",
  "train_x": "train_x.npy", "train_y": "train_y.npy",
  "val_x": "val_x.npy", "val_y": "val_y.npy",
  "test_x": "test_x.npy", "test_y": "test_y.npy",
  "train_marker_y": "train_marker_y.npy",
  "val_marker_y": "val_marker_y.npy",
  "test_marker_y": "test_marker_y.npy",
  "ghi_threshold": 1.0,

  "hist_len": 96,
  "fut_len": 96,
  "pred_len": 96,

  # ---- MMK 必須引数（ここが欠けて落ちてた） ----
  "var_num": 8,
  "hidden_dim": 64,
  "layer_num": 3,
  "layer_type": "MoK",
  "layer_hp": [["TaylorKAN", 4], ["TaylorKAN", 4], ["JacobiKAN", 4], ["JacobiKAN", 4]],

  "use_time_feature": True,
  "norm_time_feature": True,
  "time_feature_cls": ["tod", "dow"],

  # ---- 学習設定（exp_runner/traingraph 両対応） ----
  "batch_size": 64,
  "lr": 5e-5,
  "optimizer_weight_decay": 1e-4,

  "max_epochs": 120,
  "val_metric": "val/loss",
  "enable_early_stop": True,
  "early_stop_patience": 20,
  "early_stop_min_delta": 0.0,
  "gradient_clip_val": 1.0,

  "lr_step_size": 20,
  "lr_gamma": 0.5,

  "ckpt_save_top_k": 5,
  "ckpt_save_last": True,
  "ckpt_every_n_epochs": 1,
}
