# 太陽光発電量予測 — MMK-FusionPV

> **Multi-basis Mixture of KAN-experts for Solar PV Power Forecasting**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

---

## 🚀 概要 (Abstract)

本研究は、太陽光発電量の**96時間先予測**において、解釈性と予測精度を両立するモデル
**MMK-FusionPV（Multi-basis Mixture of KAN-experts）** を提案・実装したものです。

従来主流のTransformerベースの手法（PatchTST, iTransformerなど）は高い予測性能を持つ一方で、
モデル内部がブラックボックス化するという課題がありました。

MMK-FusionPVは、Kolmogorov-Arnold Network（KAN）を
複数の基底関数（B-Spline, Taylor, Fourier等）を持つエキスパート混合（Mixture of Experts）構造で構築し、
**どの気象特徴量・時間帯でどのエキスパートが機能したかを可視化・解釈可能**にしています。

---

## ✨ 主な貢献 (Key Contributions)

- **KANエキスパート混合アーキテクチャ**: 複数の非線形基底関数を切り替えミックスするMoEなKAN層の設計
- **解釈性の実証**: Taylor展開による学習済み写像の数式化、ゲートネットワーク（MoE）の寄与度ヒートマップ可視化
- **太陽光発電特化の損失設計**: 夜間の疑似ゼロ出力に対応した昼間加重MSE損失（PeakWeightedMSE）
- **再現可能な実験設計**: config駆動の実験管理、評価・可視化・解釈スクリプトの整備

---

## 🗂️ ブランチ構成

| ブランチ | 内容 |
|---|---|
| `main` | 最終モデルの学習・評価を再現するための最小構成（本ブランチ） |
| `legacy` | 実験過程のスクリプト・中間結果・旧モデル版（参照専用） |

実験の経緯や試行錯誤の詳細は `git checkout legacy` で確認できます。

---

## 🛠️ 環境構築 (Installation)

```bash
# Python 3.10 以上を推奨
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

---

## 📊 データセット (Dataset)

前処理済み NumPy データは容量の都合上このリポジトリに含まれていません。  
入手方法・配置手順は [`data/prepared/96-96_fusionA11_96_96_stride1/README.md`](data/prepared/96-96_fusionA11_96_96_stride1/README.md) を参照してください。

- **入力**: 96時間 × 12特徴量（発電量・気温・湿度・日射量・時刻エンコード・ラグ特徴）
- **出力**: 96時間先の PV発電量（kWh）
- **データ仕様**: `data/prepared/.../meta.json` を参照

---

## 💻 実行手順 (Usage)

### 学習 (Train)

```bash
# MMK-FusionPV（提案モデル）
python scripts/train_mmk.py \
  -c config/tsukuba_conf/MMK_FusionPV_FeatureToken_A11_stride1_v2.py \
  -s save --devices 1 --seed 2025

# ベースライン: iTransformer
python scripts/train_mmk.py \
  -c config/tsukuba_conf/iTransformer_peak_96_fusionA11.py \
  -s save --devices 1 --seed 2025

# ベースライン: NLinear, PatchTST, RLinear
python scripts/train_mmk.py \
  -c config/tsukuba_conf/baselines/NLinear_Tsukuba_96for96.py \
  -s save --devices 1 --seed 2025
```

### 評価 (Evaluation)

```bash
python tools/eval/eval_mmk_from_ckpt.py \
  -c config/tsukuba_conf/MMK_FusionPV_FeatureToken_A11_stride1_v2.py \
  --ckpt checkpoints/best.ckpt \
  --device cpu \
  --outdir results_eval
```

出力: `results_eval/metrics_eval.json`（MAE, MSE, RMSE, nRMSE, Day-MAE 等）

### 🔍 解釈性・可視化 (Interpretability)

MMK-FusionPVの最大の強みは、学習済みモデルの意思決定を後から解析できる点です。

#### ゲートネットワーク（エキスパート寄与度）ヒートマップ

各特徴量トークンに対してどのKANエキスパートがアクティブになっているかを可視化します。

```bash
python tools/interpret/make_gate_heatmap_featuretoken.py \
  -c config/tsukuba_conf/MMK_FusionPV_FeatureToken_A11_stride1_v2.py \
  --ckpt checkpoints/best.ckpt \
  --outdir results_gate
```

#### Taylor展開による非線形応答の数式化

KAN層が学習した各入力から出力への写像を、TaylorKANの係数から解析的な多項式として抽出します。

```bash
python tools/interpret/extract_taylorkan_formulas.py \
  -c config/tsukuba_conf/MMK_FusionPV_FeatureToken_A11_stride1_v2.py \
  --ckpt checkpoints/best.ckpt \
  --outdir results_formula
```

#### 入力応答曲線

```bash
python tools/interpret/plot_input_response_curve.py \
  -c config/tsukuba_conf/MMK_FusionPV_FeatureToken_A11_stride1_v2.py \
  --ckpt checkpoints/best.ckpt
```

---

## 📈 実験結果 (Results)

データセット: 筑波大学PVデータ（テスト期間: 2022年〜）  
評価指標: MAE（W/h、昼間のみ評価）

| モデル | MAE | MSE |
|---|---|---|
| **MMK-FusionPV (Ours)** | **9.35** | **282.68** |
| iTransformer | 8.49 | 275.42 |

> ⚠️ 上記は検証データへの評価値の一例です。学習済み重みの再現値は [`checkpoints/`](checkpoints/) に配置予定。
> 詳細な結果は `results_eval/` ディレクトリを参照（ckpt再現後に生成されます）。

---

## 🗂️ ディレクトリ構成

```
.
├── scripts/              # 実行スクリプト（train / prepare_dataset）
├── config/               # 実験設定ファイル
│   └── tsukuba_conf/     # 筑波大学データセット向け設定
│       └── baselines/    # ベースライン設定（NLinear, PatchTST, RLinear）
├── easytsf/              # モデルライブラリ
│   ├── layer/            # KAN層・Transformer層・埋め込み層
│   └── model/            # モデル実装（MMK, iTransformer等）
├── tools/
│   ├── eval/             # 評価スクリプト
│   ├── plot/             # 可視化スクリプト
│   └── interpret/        # 解釈性解析スクリプト（KAN特有機能）
├── data/                 # データ仕様（NPYは除外・meta.jsonのみ管理）
├── checkpoints/          # 学習済み重み（実体は外部配布）
└── figures/              # 論文・発表用代表図
```

---

## 📄 引用 (Citation)

本研究の成果を利用する場合は、以下を引用してください（論文投稿後に更新予定）：

```bibtex
@misc{mmkfusionpv2026,
  title  = {Multi-basis Mixture of KAN-experts for Solar PV Power Forecasting},
  author = {（著者名）},
  year   = {2026},
}
```

---

## ⚖️ ライセンス (License)

MIT License — 詳細は [LICENSE](LICENSE) を参照してください。
