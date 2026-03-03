# -*- coding: utf-8 -*-
# 既存の 96→96 天気込み設定を継承し、time特徴(+4)を使う版
# - data_dir を timeF 版に変更
# - enc_in を +4（sin/cos(hour), sin/cos(doy) を追加したため）
# - ついでに損失をピークに敏感な MSE に切替
from config.tsukuba_conf.iTransformer_Tsukuba_96for96_weather import exp_conf as _base

exp_conf = _base.copy()

# 実験名を分かりやすく
exp_conf["exp_name"] = "iTransformer_Tsukuba_96for96_weather_timeF"

# 新しい split（先ほど作成した timeF データ）を使う
# 既存設定で data_dir / data_path のどちらを見ていても上書きできるように両方指定
exp_conf["data_dir"]  = r"dataset_PV/splits_96_96_weather_timeF"
exp_conf["data_path"] = r"dataset_PV/splits_96_96_weather_timeF"

# 入力次元を +4（23 → 27）
base_enc_in = _base.get("enc_in", 23)
exp_conf["enc_in"] = int(base_enc_in) + 4

# （任意）ピーク重視にしたいので損失を MSE に
# 環境により 'criterion' or 'loss' 等のキー名が異なる場合があるため両対応で上書き
exp_conf["criterion"] = "mse"
exp_conf["loss"] = "mse"

# そのほかは既存設定のまま（学習率・epoch 等は元の値を使用）
