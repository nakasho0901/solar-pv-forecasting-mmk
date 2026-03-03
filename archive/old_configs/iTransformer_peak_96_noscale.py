# config/tsukuba_conf/iTransformer_peak_96_noscale.py
exp_conf = dict(
    # ★ここを自分のnoscaleデータに合わせる
    data_dir="dataset_PV/prepared/96-96_itr_noscale",
    train_x="train_x.npy", train_y="train_y.npy",
    val_x="val_x.npy", val_y="val_y.npy",
    test_x="test_x.npy", test_y="test_y.npy",
    train_marker_y="train_marker_y.npy",
    val_marker_y="val_marker_y.npy",
    test_marker_y="test_marker_y.npy",
    train_t0="train_t0.npy",
    val_t0="val_t0.npy",
    test_t0="test_t0.npy",

    ghi_threshold=1.0,
    hist_len=96, fut_len=96, pred_len=96,

    model_name="iTransformerPeak",

    # ===== iTransformer 本体 =====
    d_model=192,
    n_heads=8,
    e_layers=4,
    d_ff=768,
    factor=1,
    dropout=0.1,
    activation="gelu",
    output_attention=False,

    # ===== runner側の設定（重要）=====
    pv_index=0,
    ghi_index=3,
    ghi_boost=1.2,   # 1.0〜1.3の範囲で調整（効くことが多い）
    batch_size=64,

    # ===== 学習 =====
    optimizer="adam",
    lr=1e-4,
    optimizer_weight_decay=1e-4,
    lr_scheduler="step",
    lr_step_size=15,
    lr_gamma=0.5,

    max_epochs=60,
    val_metric="val/loss",
    enable_early_stop=True,
    early_stop_patience=15,
    early_stop_min_delta=0.0,
    gradient_clip_val=1.0,
)
