# config/tsukuba_conf/MMK_Mix_PV_24.py 完全版（24in/24out 用）
exp_conf = {
  "model_name": "MMK_Mix",
  "dataset_name": "Tsukuba",

  # 24in/24out の窓切り済み .npy を保存したフォルダに変更（ここが重要）
  "data_dir": "dataset_PV/processed_24_hardzero",

  "train_x": "train_x.npy", "train_y": "train_y.npy",
  "val_x": "val_x.npy", "val_y": "val_y.npy",
  "test_x": "test_x.npy", "test_y": "test_y.npy",

  # marker_y はあなたのデータ生成ログで存在が確認できている
  "train_marker_y": "train_marker_y.npy",
  "val_marker_y": "val_marker_y.npy",
  "test_marker_y": "test_marker_y.npy",

  "ghi_threshold": 1.0,

  # --- 24時間予測設定 ---
  "hist_len": 24,
  "fut_len": 24,
  "pred_len": 24,

  # 入力は8変数（x: (S, 24, 8)）
  "var_num": 8,

  # 入力長に合わせた隠れ次元（96版は96、24版は24）
  "hidden_dim": 24,

  "layer_num": 3,
  "layer_type": "MoK",
  "layer_hp": [
      ["KAN", 5],
      ["WavKAN", 5],
      ["TaylorKAN", 4],
      ["JacobiKAN", 4],
  ],

  "use_time_feature": True,
  "norm_time_feature": True,
  "time_feature_cls": ["tod", "dow"],

  "batch_size": 64,

  # --- 学習設定（今までと同じ） ---
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

  "enforce_nonneg": True,
  "force_night0": True,
  "use_marker_mask": True,
}
