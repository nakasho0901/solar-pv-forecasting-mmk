# -*- coding: utf-8 -*-
"""
学習タスクの共通設定。
注意:
- ここでの max_epochs が 10 だと、個別設定(exp_conf)よりも優先されて 10epoch で打ち切られるケースがあります。
- 再現性のため、上限を十分大きくし、EarlyStopping(es_patience)で止めるのが安全です。
- batch_size もここで大きすぎるとメモリや DataLoader がボトルネックになるので、32 をデフォルトにしています。
"""

task_conf = dict(
    # 入出力長（データ分割と合わせる）
    hist_len=96,
    pred_len=96,

    # ====== DataLoader ======
    batch_size=32,          # 64→32 に下げて安定化（環境次第で調整）
    num_workers=2,

    # ====== 学習スケジュール ======
    # ★ 10 で固定されていたのが早期打ち切りの原因。
    #    ここは大きめ(=50)にして、EarlyStopping で止める運用が安全。
    max_epochs=50,          # 10→50
    es_patience=8,          # 10→8（汎化の谷を拾いつつ長すぎない程度）

    # ====== 最適化 ======
    lr=1e-4,                # 個別モデル側の設定があるならそちらが優先される想定
    optimizer="AdamW",
    optimizer_betas=(0.95, 0.9),
    optimizer_weight_decay=1e-5,

    # ====== LR Scheduler ======
    lr_scheduler='StepLR',
    lr_step_size=1,
    lr_gamma=0.5,

    # ====== クリップ ======
    gradient_clip_val=5,

    # ====== 評価指標（ログ名） ======
    val_metric="val/loss",
    test_metric="test/mae",

    # ====== 時刻特徴の扱い ======
    norm_time_feature=False,
    time_feature_cls=["tod", "dow"],
)
