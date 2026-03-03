# config/tsukuba_conf/MMK_Mix_PV_24_pvin.py
# =========================================================
# 24-24 / PV-input / MMK_Mix 専用 config（確定版）
# =========================================================

exp_conf = {
    # ===== model =====
    # ★ traingraph_24_24_pvin.py は MMK_Mix_24 しか受け付けない
    "model_name": "MMK_Mix_24",

    # ===== dataset (NPYDataInterface) =====
    "data_dir": r"dataset_PV/processed_new/24-24_pvin",

    "train_x": "train_x.npy",
    "train_y": "train_y.npy",
    "train_marker_y": "train_daymask_y.npy",

    "val_x": "val_x.npy",
    "val_y": "val_y.npy",
    "val_marker_y": "val_daymask_y.npy",

    "test_x": "test_x.npy",
    "test_y": "test_y.npy",
    "test_marker_y": "test_daymask_y.npy",

    # ===== window =====
    "hist_len": 24,
    "pred_len": 24,
    "fut_len": 24,   # 互換用（参照されても問題ないように）

    # ===== variables =====
    # ★ あなたのログから確定した値
    "var_num": 23,
    "pv_index": 0,    # pv_kwh が feature の先頭
    "ghi_index": 3,   # ghi_wm2

    # ===== daymask (24-24用) =====
    "use_input_daymask": True,     # marker_y ではなく var_x から daymask を作る
    "daylight_feature_index": 8,   # feature 内の is_daylight の位置（0始まり）


    # ===== MMK(MoK) structure =====
    "hidden_dim": 48,
    "layer_num": 3,
    "layer_hp": [
        ["KAN", 5],
        ["WavKAN", 5],
        ["TaylorKAN", 4],
        ["JacobiKAN", 4],
    ],

    # ===== training =====
    "batch_size": 64,
    "max_epochs": 30,

    "lr": 1e-4,
    "optimizer_weight_decay": 1e-4,

    "enable_early_stop": True,
    "early_stop_patience": 10,
    "gradient_clip_val": 1.0,

    # ===== scheduler =====
    "lr_step_size": 15,
    "lr_gamma": 0.5,

    # ===== loss / mask =====
    # traingraph_24_24_pvin.py 側で
    # ・出力から pv_index を抽出
    # ・daymask で昼のみ loss
    "use_daymask_loss": True,

    # ===== options =====
    "ghi_boost": 1.0,  # 24-24 では OFF 推奨
}
