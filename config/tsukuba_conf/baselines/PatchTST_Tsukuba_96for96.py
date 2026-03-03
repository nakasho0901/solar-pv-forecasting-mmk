# config/tsukuba_conf/PatchTST_Tsukuba_96for96.py
exp_conf = dict(
    model_name="PatchTST",
    dataset_name="Tsukuba",

    hist_len=96,
    pred_len=96,

    patch_len=16,
    stride=8,
    output_attention=False,
    d_model=256,
    d_ff=256,            # ← 追加
    dropout=0.2,
    factor=3,
    n_heads=8,
    activation="gelu",
    e_layers=3,

    lr=0.001,
    max_epochs=20,
    batch_size=128,
)
