# config/tsukuba_conf/MMK_Tsukuba_96for96.py
exp_conf = dict(
    model_name="MMK",
    dataset_name="Tsukuba",

    # 入出力長
    hist_len=96,
    pred_len=96,

    # モデル構成
    hidden_dim=64,
    layer_type="MoK",
    layer_hp=[['TaylorKAN', 4], ['TaylorKAN', 4], ['JacobiKAN', 4], ['JacobiKAN', 4]],
    layer_num=3,

    # 学習設定（LRはスイープ最良）
    lr=0.0003,
    max_epochs=20,
    batch_size=32,

    # ★ 時刻特徴をMMKで使う（先に MMK.py を差し替えてある前提）
    use_time_feature=True,

    # （任意）ローダ側の時間特徴指定：tod=時刻, dow=曜日
    norm_time_feature=True,
    time_feature_cls=["tod", "dow"],
)
