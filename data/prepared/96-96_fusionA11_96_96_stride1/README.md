# データセット: 96-96_fusionA11_96_96_stride1

## 概要

太陽光発電量の96時間予測タスク用に前処理済みのデータセットです。

| 項目 | 値 |
|---|---|
| 入力長 | 96時間 |
| 予測長 | 96時間 |
| stride | 1時間 |
| 特徴量数 | 12（`meta.json` の `feature_cols` 参照） |
| train サンプル数 | 19,726 |
| val サンプル数 | 2,818 |
| test サンプル数 | 5,636 |

---

## ファイル構成

| ファイル名 | 形状 | 説明 |
|---|---|---|
| `train_x.npy` | (19726, 96, 12) | 学習用入力（12特徴量） |
| `train_y.npy` | (19726, 96) | 学習用目標（pv_kwh） |
| `train_marker_y.npy` | (19726, 96) | 学習用昼間マスク（0=夜, 1=昼） |
| `train_t0.npy` | (19726,) | 各ウィンドウ開始時刻（Unix秒） |
| `val_x.npy` | (2818, 96, 12) | 検証用入力 |
| `val_y.npy` | (2818, 96) | 検証用目標 |
| `val_marker_y.npy` | (2818, 96) | 検証用昼間マスク |
| `val_t0.npy` | (2818,) | 検証用ウィンドウ開始時刻 |
| `test_x.npy` | (5636, 96, 12) | テスト用入力 |
| `test_y.npy` | (5636, 96) | テスト用目標 |
| `test_marker_y.npy` | (5636, 96) | テスト用昼間マスク |
| `test_t0.npy` | (5636,) | テスト用ウィンドウ開始時刻 |
| `meta.json` | — | データ仕様書（特徴量名・分割比等）← Git管理対象 |

> NPYファイルは容量の都合上 `.gitignore` で除外されています。  
> 以下の手順でデータを入手・配置してください。

---

## データの入手方法

<!-- ★ 配布先URLまたは連絡先を記入してください -->

- **研究室内共有**: （担当者に連絡してください）
- **外部公開**: 準備中（Zenodo DOI: 未発行）

---

## ファイルの配置場所

ダウンロード後、以下のパスに展開してください：

```
data/
└── prepared/
    └── 96-96_fusionA11_96_96_stride1/
        ├── train_x.npy
        ├── train_y.npy
        ├── train_marker_y.npy
        ├── train_t0.npy
        ├── val_x.npy
        ├── val_y.npy
        ├── val_marker_y.npy
        ├── val_t0.npy
        ├── test_x.npy
        ├── test_y.npy
        ├── test_marker_y.npy
        ├── test_t0.npy
        └── meta.json  ← すでにリポジトリに含まれています
```

---

## 前処理の再現（スクリプトから生成する場合）

元の生データ（秒単位PV計測値・気象データ）から前処理を再現する場合：

```bash
python scripts/prepare_dataset.py \
  --raw_dir <生データのディレクトリ> \
  --out_dir data/prepared/96-96_fusionA11_96_96_stride1
```

元の生データは `dataset_PV/4 Summarized_data_per_second.csv`（秒単位発電量）と
`dataset_PV/3 Raw hourly data solar irradiation yyyymmdd.csv`（気象データ）です。

---

## 特徴量の説明

`meta.json` の `feature_cols` に対応する12特徴量：

| インデックス | 特徴量名 | 説明 |
|---|---|---|
| 0 | `pv_kwh` | 太陽光発電量 kWh（予測ターゲット） |
| 1 | `temp_c` | 気温 °C |
| 2 | `rh_pct` | 相対湿度 % |
| 3 | `ghi_wm2` | 全天日射量 W/m² |
| 4 | `hour_sin` | 時刻の sin エンコード |
| 5 | `hour_cos` | 時刻の cos エンコード |
| 6 | `is_daylight` | 昼間フラグ（0/1） |
| 7 | `lag_pv_kwh_24h` | 24時間前の発電量 |
| 8 | `lag_ghi_wm2_1h` | 1時間前の日射量 |
| 9 | `lag_ghi_wm2_2h` | 2時間前の日射量 |
| 10 | `lag_ghi_wm2_3h` | 3時間前の日射量 |
| 11 | `lag_ghi_wm2_24h` | 24時間前の日射量 |

スケーリングなし（RevIN をモデル内部で適用）。
