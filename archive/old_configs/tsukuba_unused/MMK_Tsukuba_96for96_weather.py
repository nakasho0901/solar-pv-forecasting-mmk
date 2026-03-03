# -*- coding: utf-8 -*-
# 実行: python train.py -c config/tsukuba_conf/MMK_Tsukuba_96for96_weather.py
exp_conf = {
    "model_id": "MMK_Tsukuba_96for96_weather",
    "model": "MMK",
    "model_name": "MMK",
    "dataset_name": "Tsukuba",

    # ===== Data =====
    "data": "custom",
    "data_path": "dataset_PV/splits_96_96_weather",

    # ===== IO dims =====
    "features": "M",
    "seq_len": 96,
    "label_len": 48,
    "pred_len": 96,
    "enc_in": 23,
    "dec_in": 23,
    "c_out": 1,
    "var_num": 1,

    # ===== Train (train.py が読むキー名) =====
    "batch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 1e-2,
    "max_epochs": 50,      # ★ ここが効く
    "es_patience": 8,      # ★ ここが効く
    "grad_clip": 1.0,
    "seed": 1,
    "devices": "0",
    "use_wandb": False,
    "save_root": "save",

    # ===== MMK hyperparams =====
    "hidden_dim": 256,
    "layer_type": "TaylorKAN",
    "layer_hp": 3,
    "layer_num": 3,
    "dropout": 0.10,
    "activation": "gelu",
    "output_attention": False,
}
