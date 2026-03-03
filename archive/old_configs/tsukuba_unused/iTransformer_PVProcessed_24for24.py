"""
設定ファイル: config/tsukuba_conf/iTransformer_PVProcessed_24for24.py
PVProcessed (24→24) 用 iTransformer 設定

※ 前の 96→96 用
   iTransformer_PVProcessed_96for96.py をベースに、
   ・hist_len / pred_len を 24 に変更
   ・data_dir を 24→24 用の NPY データフォルダに変更
"""

# ===============================
#  実験設定 (train.py から読み込まれる)
# ===============================

exp_conf = dict(
    # ===================== データ設定 =====================
    # データ形式: 今回は窓切り済み .npy を使用
    data_format     = "npy",

    # NPYDataInterface 用:
    #   train_x.npy / val_x.npy / test_x.npy などが置いてあるディレクトリ
    #   → 24→24 用に作った窓切りデータのフォルダに合わせてください
    #      例: dataset_PV/processed_24for24/train_x.npy など
    data_dir        = "dataset_PV/processed_24for24",

    # LTSFRunner 用:
    #   {dataset_name}.npz (= Tsukuba.npz) を読むルート
    #   正規化パラメータは 96→96 と共通で良いので、
    #   Tsukuba.npz が置いてある元のフォルダを指す
    data_root       = "dataset_PV/processed",

    # Tsukuba.npz のファイル名のベース（{dataset_name}.npz）
    dataset_name    = "Tsukuba",

    # 入力 24[h], 予測 24[h]
    hist_len        = 24,   # ← ここを 24 に変更
    pred_len        = 24,   # ← ここを 24 に変更

    # 目的変数の列インデックス (PV 列)
    target_col_idx  = 0,

    # DataLoader 共通
    batch_size      = 64,
    num_workers     = 4,

    # ===================== モデル設定 (iTransformer) =====================
    model_name      = "iTransformer",

    # 変数数（特徴量数）
    #   train_x.npy の shape: (N, 24, 8) を想定
    enc_in          = 8,      # 入力系列のチャネル数
    dec_in          = 8,      # （iTransformer ではほぼ enc_in と同じ扱い）
    c_out           = 1,      # 出力は PV 1 本

    # iTransformer 固有ハイパーパラメータ
    d_model         = 256,
    n_heads         = 8,
    d_ff            = 512,
    e_layers        = 2,
    dropout         = 0.1,
    activation      = "gelu",
    factor          = 3,
    output_attention = False,

    # ===================== 学習設定 =====================
    max_epochs      = 30,

    # Optimizer
    optimizer       = "AdamW",
    lr              = 1e-3,     # ※ 96→96 と同じ。tuning したい場合はここを変える
    optimizer_weight_decay = 1e-4,

    # LR Scheduler
    lr_scheduler    = "StepLR",
    lr_step_size    = 5,
    lr_gamma        = 0.5,

    # EarlyStopping / ReduceLROnPlateau で監視する指標名
    val_metric      = "val/loss",
)
