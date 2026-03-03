# config/tsukuba_conf/RLinear_Tsukuba_96for96.py
exp_conf = dict(
    model_name="RLinear",
    dataset_name="Tsukuba",
    hist_len=96,
    pred_len=96,
    lr=0.001,
    max_epochs=10,
    batch_size=128,
)
