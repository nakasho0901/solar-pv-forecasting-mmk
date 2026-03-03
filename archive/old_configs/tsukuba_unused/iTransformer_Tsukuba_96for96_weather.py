# -*- coding: utf-8 -*-
# 実行: python train.py -c config/tsukuba_conf/iTransformer_Tsukuba_96for96_weather.py
exp_conf = {
    "model_id": "iTransformer_Tsukuba_96for96_weather",
    "model": "iTransformer",
    "model_name": "iTransformer",
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

    # ===== Train (train.py が読むキー名) =====
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "max_epochs": 50,      # ★ train.py はこれを使う
    "es_patience": 8,      # ★ train.py はこれを使う
    "grad_clip": 1.0,
    "seed": 1,
    "devices": "0",
    "use_wandb": False,
    "save_root": "save",

    # ===== iTransformer hyperparams =====
    "d_model": 256,
    "n_heads": 8,
    "e_layers": 3,
    "d_ff": 512,
    "dropout": 0.1,
    "factor": 3,
    "activation": "gelu",
    "output_attention": False,
}
