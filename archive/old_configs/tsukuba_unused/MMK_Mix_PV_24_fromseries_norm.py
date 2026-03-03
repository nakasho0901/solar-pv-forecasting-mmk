# config/tsukuba_conf/MMK_Mix_PV_24_fromseries_norm_huber.py
# -*- coding: utf-8 -*-

"""
MMK_Mix_PV_24_fromseries_norm_huber.py
- 24in/24out (fromseries) の正規化データセット用 config
- 損失を Huber (SmoothL1) に変更して「平均解」に落ちにくくする
- 単位：学習中は正規化（無次元）、可視化・評価時に kWh に戻す（y_max で逆正規化）

使い方（例）
python traingraph_24.py -c config\\tsukuba_conf\\MMK_Mix_PV_24_fromseries_norm_huber.py -s save\\MMK_Mix_PV_24_fromseries_norm_huber_v1 --devices 1 --seed 2025
"""

import torch

# traingraph_24.py が import する想定
exp_conf = {
    # ===== 基本 =====
    "model_name": "MMK_Mix",
    "dataset_name": "NPYDataInterface",

    # ===== データ =====
    "data_dir": "dataset_PV/processed_24_fromseries_hardzero_v2_norm",

    "train_x": "train_x.npy",
    "train_y": "train_y.npy",
    "val_x": "val_x.npy",
    "val_y": "val_y.npy",
    "test_x": "test_x.npy",
    "test_y": "test_y.npy",

    "train_marker_y": "train_marker_y.npy",
    "val_marker_y": "val_marker_y.npy",
    "test_marker_y": "test_marker_y.npy",

    # ===== 形状 =====
    "hist_len": 24,
    "fut_len": 24,
    "pred_len": 24,

    # var_x の特徴量数（※time feature をモデルに入れるなら後で増やす必要あり）
    "var_num": 8,

    # ===== MMK_Mix ハイパラ =====
    "hidden_dim": 96,
    "layer_num": 3,
    "layer_type": "MoK",
    "layer_hp": [
        ["KAN", 5],
        ["WavKAN", 5],
        ["TaylorKAN", 4],
        ["JacobiKAN", 4],
    ],

    # ===== time feature（runner側が作る想定）=====
    # ※現状の MMK_Mix.forward が marker_x を使っていない疑いが強いので
    #   まずは設定は残す（後でモデル側を完全版で修正する）
    "use_time_feature": True,
    "norm_time_feature": True,
    "time_feature_cls": ["tod", "dow"],

    # ===== 学習 =====
    "batch_size": 64,
    "lr": 1e-4,
    "optimizer_weight_decay": 1e-4,
    "max_epochs": 30,

    # 早期終了
    "early_stop_patience": 8,
    "early_stop_min_delta": 1e-4,

    # ===== 物理制約（※評価側・学習側で適用されるかは traingraph_24 の実装次第）=====
    "enforce_nonneg": True,
    "force_night0": True,
    "use_marker_mask": True,

    # ===== index（重要：night判定や特徴量処理に使う）=====
    "ghi_index": 2,
    "pv_index": 0,
    "ghi_boost": 1.0,

    # ===== 損失：Huber =====
    # traingraph_24.py が "loss_fn" を読む作りならこれで確実に Huber になる
    "loss_fn": torch.nn.SmoothL1Loss(beta=0.1),
}
