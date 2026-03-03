# config/tsukuba_conf/MMK_Mix_PV_96.py 完全版
exp_conf = {
  "model_name": "MMK_Mix",
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

  "var_num": 8,
  # 隠れ層の次元を拡張し、複雑な気象パターンの学習能力を向上
  "hidden_dim": 96,  
  "layer_num": 3,     
  "layer_type": "MoK",
  "layer_hp": [
      ["KAN", 5], ["WavKAN", 5], ["TaylorKAN", 4], ["JacobiKAN", 4]
  ],

  "use_time_feature": True,
  "norm_time_feature": True,
  "time_feature_cls": ["tod", "dow"],

  "batch_size": 64,
  # --- 全体誤差最小化のための「最適化」学習戦略 ---
  "lr": 1e-4,                      # 山を登り切るパワーを確保するため微増
  "optimizer_weight_decay": 1e-4,  # 過学習を抑え、未知データへの汎化性能を高める

  "max_epochs": 30,                # じっくりと誤差を削るための十分な試行回数
  "val_metric": "val/loss",
  "enable_early_stop": True,
  "early_stop_patience": 8,       # 専門家(Expert)が最適な役割分担を見つけるのを待つ
  "early_stop_min_delta": 0.0001,
  "gradient_clip_val": 1.0,

  "lr_step_size": 15,              # 15エポックごとに学習率を下げて、精密に収束させる
  "lr_gamma": 0.5,

  "ckpt_save_top_k": 5,
  "ckpt_save_last": True,
  
  "enforce_nonneg": True,
  "force_night0": True,
  "use_marker_mask": True,
}