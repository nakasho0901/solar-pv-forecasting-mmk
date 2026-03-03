# config/tsukuba_conf/MMK_PVProcessed_96for96.py
# -*- coding: utf-8 -*-
"""
Tsukuba の窓切り済みデータ（dataset_PV/processed）を使って、
MMK モデルで 96→96 の発電量予測を行うための設定ファイル（完全修正版）。
iTransformer と同じ Optimizer / Scheduler / EarlyStopping / Logging が動くよう統一済み。
"""

exp_conf = dict(
    # =========================
    # 実験識別
    # =========================
    model_name="MMK",
    dataset_name="Tsukuba",

    # =========================
    # 入出力長・次元
    # =========================
    hist_len=96,
    pred_len=96,
    seq_len=96,     # 参照される実装があるため明示
    enc_in=8,       # 入力特徴量数（temp, rh, ghi, pv, sinH, cosH, sinD, cosD）
    var_num=8,      # ← MMK が必須としている変数数
    c_out=1,        # 予測は PV 1 系列

    # =========================
    # データ読み（窓切り済み .npy）
    # =========================
    data_format="npy",
    data_dir="dataset_PV/processed",
    train_x="train_x.npy",
    train_y="train_y.npy",
    val_x="val_x.npy",
    val_y="val_y.npy",
    test_x="test_x.npy",
    test_y="test_y.npy",

    # =========================
    # MMK 側ハイパラ
    # =========================
    hidden_dim=64,
    layer_num=3,
    layer_type="MoK",
    layer_hp=[['TaylorKAN', 4], ['TaylorKAN', 4], ['JacobiKAN', 4], ['JacobiKAN', 4]],
    use_time_feature=True,
    norm_time_feature=True,
    time_feature_cls=["tod", "dow"],

    # =========================
    # 学習ハイパラ（エラーが出ないよう iTr と完全互換化）
    # =========================
    batch_size=64,
    lr=3e-4,
    optimizer_weight_decay=1e-4,    # ★ AdamW に必須（欠けていたので追加）
    max_epochs=100,                  # 96→96 では30より高く設定しておく方が安定
    early_stop_patience=10,         # ★ traingraph.py が参照（デフォルトは5）
    val_metric="val/loss",

    # ---- Scheduler の追加（これが無くて落ちていた） ----
    lr_step_size=5,                 # ★ StepLR に必須
    lr_gamma=0.5,                   # ★ StepLR に必須

    # Gradient clipping
    gradient_clip_val=1.0,
    precision="32-true",

    # =========================
    # ログ・再現・デバイス
    # =========================
    save_root="save",
    seed=2025,
    use_wandb=False,
    devices=1,

    # =========================
    # 目的列（PV）
    # =========================
    target_col_idx=3,
)
