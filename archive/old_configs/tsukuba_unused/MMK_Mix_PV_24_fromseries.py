# config/tsukuba_conf/MMK_Mix_PV_24_fromseries.py
exp_conf = {
    "model_name": "MMK_Mix",
    "dataset_name": "Tsukuba",

    # ★ 連続CSVから作った24hデータ
    "data_dir": "dataset_PV/processed_24_fromseries_hardzero_v2",

    "train_x": "train_x.npy",
    "train_y": "train_y.npy",
    "val_x": "val_x.npy",
    "val_y": "val_y.npy",
    "test_x": "test_x.npy",
    "test_y": "test_y.npy",

    "train_marker_y": "train_marker_y.npy",
    "val_marker_y": "val_marker_y.npy",
    "test_marker_y": "test_marker_y.npy",

    # ===== 24時間設定 =====
    "hist_len": 24,
    "fut_len": 24,
    "pred_len": 24,

    # 入力次元（ログと一致）
    "var_num": 8,

    # ===== モデル構造（96hと同じ思想）=====
    "hidden_dim": 96,
    "layer_num": 3,
    "layer_type": "MoK",
    "layer_hp": [
        ["KAN", 5],
        ["WavKAN", 5],
        ["TaylorKAN", 4],
        ["JacobiKAN", 4]
    ],

    # 時刻特徴
    "use_time_feature": True,
    "norm_time_feature": True,
    "time_feature_cls": ["tod", "dow"],

    # 学習設定
    "batch_size": 64,
    "lr": 1e-4,
    "optimizer_weight_decay": 1e-4,
    "max_epochs": 30,

    # Early stopping
    "early_stop_patience": 8,
    "early_stop_min_delta": 1e-4,

    # 物理制約
    "enforce_nonneg": True,
    "force_night0": True,
    "use_marker_mask": True,

    # ===== traingraph 用（重要）=====
    "ghi_index": 2,   # x[:, :, 2] = ghi_wm2
    "pv_index": 0,    # 出力が (B,T,1) なので 0
    "ghi_boost": 1.3,
}
