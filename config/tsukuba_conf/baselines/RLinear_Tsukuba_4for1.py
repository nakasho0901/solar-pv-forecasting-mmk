# config/tsukuba_conf/RLinear_Tsukuba_4for1.py
exp_conf = dict(
    model_name="RLinear",
    dataset_name='Tsukuba',

    hist_len=4,   # 過去4点を見る
    pred_len=1,   # 1点先を予測

    lr=0.001,
    max_epochs=2,
    batch_size=1,
)
