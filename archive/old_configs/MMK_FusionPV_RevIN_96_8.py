# ============================================================
# MMK + RevIN (96 -> 96, FusionPV A案9特徴量) / 新データ用
# - model は easytsf/model/MMK_Mix.py（あなたが貼った版）を使用
# - layer_hp は「旧4基底 list形式」でも「dict形式」でもOK
# ============================================================

USE_4BASIS_LIST = True  # True: 旧4基底(list) / False: 従来(dict)

exp_conf = {
    # -----------------------------
    # 基本
    # -----------------------------
    "model_name": "MMK_Mix",

    # -----------------------------
    # データ（A案9特徴量）
    # -----------------------------
    "data_dir": r"dataset_PV\prepared\96-96_fusionA9_96_96_stride4",

    "hist_len": 96,
    "pred_len": 96,

    # ★重要：9特徴量
    "var_num": 9,

    # 窓切りログより固定
    "pv_index": 0,
    "ghi_index": 3,
    "ghi_boost": 1.0,

    # -----------------------------
    # モデル（MMK_Mix）
    # -----------------------------
    "hidden_dim": 64,
    "layer_num": 5,
    "use_norm": True,

    # -----------------------------
    # expert 設定
    # -----------------------------
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
            "kan_hp": {
                "layer_type": "KAN",
                "hyperparam": 5,
            }
        }
    ),

    # -----------------------------
    # RevIN（MMK_Mix 側が対応済み）
    # -----------------------------
    "use_revin": True,
    "revin_eps": 1e-5,
    "revin_affine": True,

    # -----------------------------
    # 学習
    # -----------------------------
    "max_epochs": 30,
    "batch_size": 64,
    "num_workers": 0,
    "lr": 1e-4,

    # -----------------------------
    # EarlyStopping / 保存
    # -----------------------------
    "early_stop_patience": 8,
    "save_best": True,
    "save_last": True,
    "log_every_n_steps": 10,
}
