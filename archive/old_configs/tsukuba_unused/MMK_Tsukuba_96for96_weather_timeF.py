# -*- coding: utf-8 -*-
# 目的: MMK(96→96, 天気特徴) を「時間sin/cos(+4) + MSE損失」にした timeF 版へ
# 方針: 既存設定を継承し、差分だけ上書き（安全・再現性◎）

from config.tsukuba_conf.MMK_Tsukuba_96for96_weather import exp_conf as _base  # 既存MMK設定を継承

exp_conf = _base.copy()

# 実験名を区別できるように
exp_conf["exp_name"] = "MMK_Tsukuba_96for96_weather_timeF"

# ===== データ: timeF split を使う =====
# tools/add_time_features.py で作った split を指す
exp_conf["data_dir"]  = r"dataset_PV/splits_96_96_weather_timeF"
exp_conf["data_path"] = r"dataset_PV/splits_96_96_weather_timeF"

# ===== 入力次元: +4（sin/cos(hour/24), sin/cos(doy/365)）=====
base_enc_in = _base.get("enc_in", 23)   # 既存が23なら → 27
exp_conf["enc_in"] = int(base_enc_in) + 4

# ===== 損失: ピーク外しに強い罰を与える =====
# 実装により 'criterion' or 'loss' のキー名が異なる可能性があるため両方を書いておく
exp_conf["criterion"] = "mse"
exp_conf["loss"] = "mse"

# ===== 収束を少し丁寧に（振幅不足の緩和）=====
# 既存にキーがあれば上書き、無ければ無視されるので安全
exp_conf["learning_rate"] = 5e-4
exp_conf["max_epochs"]    = max(_base.get("max_epochs", 20), 50)
exp_conf["es_patience"]   = max(_base.get("es_patience", 5), 10)

# 余計な正則化で振幅が縮むのを避ける（存在する場合のみ反映）
if "weight_decay" in exp_conf:
    exp_conf["weight_decay"] = 0.0
