# config/tsukuba_conf/iTransformer_peak_96.py
exp_conf = dict(
    data_dir="dataset_PV/processed_hardzero",
    train_x="train_x.npy", train_y="train_y.npy",
    val_x="val_x.npy", val_y="val_y.npy",
    test_x="test_x.npy", test_y="test_y.npy",
    train_marker_y="train_marker_y.npy",
    val_marker_y="val_marker_y.npy",
    test_marker_y="test_marker_y.npy",
    ghi_threshold=1.0,
    hist_len=96, fut_len=96, pred_len=96,

    model_name="iTransformerPeak", # 新名称
    d_model=128, n_heads=8, e_layers=3, d_ff=512,
    factor=1, dropout=0.1, activation='gelu', output_attention=False,

    optimizer="adam",
    lr=5e-5, # runnerの期待する名称
    optimizer_weight_decay=1e-4,
    lr_scheduler="step", lr_step_size=20, lr_gamma=0.5,

    max_epochs=50,
    val_metric="val/loss",
    enable_early_stop=True,
    early_stop_patience=15,
    early_stop_min_delta=0.0,
    gradient_clip_val=1.0,
)