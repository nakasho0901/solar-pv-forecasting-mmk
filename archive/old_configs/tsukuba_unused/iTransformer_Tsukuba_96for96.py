# config/tsukuba_conf/iTransformer_Tsukuba_96for96.py
# -*- coding: utf-8 -*-
# iTransformer 96→96 (+Weather) 設定ファイル
# 使い方（1行）: python train.py -c config/tsukuba_conf/iTransformer_Tsukuba_96for96_weather.py

# 必須パラメータ
args = {
    # 識別子
    "model_id": "iTransformer_Tsukuba_96for96_weather",

    # モデル種別
    "model": "iTransformer",          # 実装側で対応しているモデル名

    # データ設定
    "data": "custom",                 # リーダが参照するデータ種別（実装に合わせて）
    "data_path": "dataset_PV/splits_96_96_weather",  # make_splits.py の出力ディレクトリ

    # 入出力次元
    "features": "M",                  # multivariate 入力を想定
    "seq_len": 96,                    # 入力長
    "label_len": 48,                  # デコーダに与える既知ラベル長（実装に合わせて）
    "pred_len": 96,                   # 予測長
    "enc_in": 23,                     # 入力特徴次元 (= X の最後の次元) ※今回のスプリット結果より
    "dec_in": 23,                     # 実装により未使用でも enc_in と揃えておく
    "c_out": 1,                       # 予測の出力次元（PVのみ）

    # 学習ハイパラ
    "batch_size": 32,
    "learning_rate": 3e-4,
    "train_epochs": 30,
    "patience": 5,                    # 早期終了の猶予（実装にある場合）
    "weight_decay": 1e-2,             # 実装にある場合
    "grad_clip": 1.0,                 # 実装にある場合（0 or None で無効）

    # 乱数・再現性
    "seed": 2025,

    # 保存・ログ
    "save_root": "save",              # 実装に合わせて使用（train.py の -s と併用可）
    "use_wandb": False,               # 実装で対応していればON/OFF
    "exp_notes": "Tsukuba PV + Weather (temp_c,rh_pct,ghi_wm2 + lags/rollings); enc_in=23",

    # デバイス（実装に依存）：例 "0" / "0,1" / "cpu"
    "devices": "0",
}

# 必要に応じて追加（実装が参照する場合）
# args.update({
#     "dropout": 0.1,
#     "d_model": 512,
#     "n_heads": 8,
#     "e_layers": 2,
#     "d_layers": 1,
# })

# 参考: 入力23次元の内訳（情報用・コードでは未使用）
# temp_c, rh_pct, ghi_wm2, hour_sin, hour_cos, dow_sin, dow_cos, is_daylight,
# lag_pv_kwh_{1,2,3,24}h, lag_ghi_wm2_{1,2,3,24}h, lag_temp_c_{1,2,3,24}h,
# roll_ghi_wm2_24h_mean, roll_temp_c_24h_mean
