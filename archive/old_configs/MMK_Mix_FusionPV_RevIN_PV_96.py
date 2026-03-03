# -*- coding: utf-8 -*-

exp_conf = {
    # ===== Dataset =====
    "data_dir": r"dataset_PV\prepared\96-96_pvin_noscale",
    "train_x": "train_x.npy", "train_y": "train_y.npy",
    "val_x": "val_x.npy", "val_y": "val_y.npy",
    "test_x": "test_x.npy", "test_y": "test_y.npy",

    # （あるなら使う：今の runner は読み込む想定）
    "train_marker_y": "train_marker_y.npy",
    "val_marker_y": "val_marker_y.npy",
    "test_marker_y": "test_marker_y.npy",

    # ===== Task =====
    "hist_len": 96,
    "pred_len": 96,

    # ===== Model =====
    "model_name": "MMK_Mix",
    "var_num": 22,
    "hidden_dim": 128,
    "layer_num": 3,
    "layer_hp": [
        ["KAN", 5],
        ["WavKAN", 5],
        ["TaylorKAN", 4],
        ["JacobiKAN", 4],
    ],

    # PV切り出し（今回の FusionPV は (B,96,1) を返す想定だけど、保険で置く）
    "pv_index": 0,

    # ===== Training =====
    "batch_size": 64,
    "lr": 1e-4,
    "optimizer_weight_decay": 1e-4,

    "max_epochs": 30,
    "enable_early_stop": True,
    "val_metric": "val/loss",
    "early_stop_patience": 15,
    "early_stop_min_delta": 0.0,
    "gradient_clip_val": 1.0,

    "lr_step_size": 20,
    "lr_gamma": 0.5,

    "ckpt_save_top_k": 5,
    "ckpt_save_last": True,

    # ===== Loss (PeakWeightedMSE) =====
    # noscale(kW)なので threshold/weight は後で調整前提（まずは無難値）
    "peak_threshold": 50.0,
    "peak_weight": 3.0,

    # ★学習中はまず False 推奨（推論・可視化で clip のほうが比較が楽）
    "enforce_nonneg": False,
}
