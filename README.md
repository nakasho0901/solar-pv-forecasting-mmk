# 太陽光発電量予測 — MMK-FusionPV

**太陽光発電量の96時間先予測**を、多基底KANエキスパート混合モデル（MMK: Multi-basis Mixture of KAN-experts）で実現する研究コードです。

---

## ブランチ構成

| ブランチ | 内容 |
|---|---|
| `main` | 最終モデルの学習・評価を再現するための最小構成 |
| `legacy` | 実験過程のスクリプト・中間結果・旧モデル版（参照用） |

`main` ブランチは再現性を優先した最小構成です。実験の経緯や試行錯誤の履歴は `legacy` ブランチに保存されています。

---

## モデル概要

- **モデル名**: MMK-FusionPV（Multi-basis Mixture of KAN-experts）
- **タスク**: 太陽光発電量の96時間先予測（96 → 96 step）
- **特徴量**: 12次元（発電量・気温・湿度・日射量・時刻エンコード・ラグ特徴）
- **ベースライン比較**: iTransformer, PatchTST, NLinear, RLinear

---

## 環境構築

```bash
# Python 3.10 以上を推奨
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

---

## データの準備

前処理済み NPY データは容量の都合上リポジトリに含まれていません。  
入手方法は [`data/prepared/96-96_fusionA11_96_96_stride1/README.md`](data/prepared/96-96_fusionA11_96_96_stride1/README.md) を参照してください。

データ仕様（特徴量・分割比）は `meta.json` に記載されています。

---

## 学習

```bash
# MMK-FusionPV（最終モデル）
python scripts/train_mmk.py \
  -c config/tsukuba_conf/MMK_FusionPV_FeatureToken_A11_stride1_v2.py \
  -s save \
  --devices 1 \
  --seed 2025

# ベースライン: iTransformer
python scripts/train_mmk.py \
  -c config/tsukuba_conf/iTransformer_peak_96_fusionA11.py \
  -s save \
  --devices 1 \
  --seed 2025
```

---

## 評価

```bash
python tools/eval/eval_mmk_from_ckpt.py \
  -c config/tsukuba_conf/MMK_FusionPV_FeatureToken_A11_stride1_v2.py \
  --ckpt checkpoints/best.ckpt \
  --device cpu \
  --outdir results_eval
```

出力: `results_eval/metrics_eval.json`（MAE, MSE, RMSE, nRMSE, MAPE 等）

---

## 結果（暫定）

| モデル | MAE | MSE | Day-MAPE |
|---|---|---|---|
| **MMK-FusionPV (Ours)** | — | — | — |
| iTransformer | — | — | — |
| PatchTST | — | — | — |
| NLinear | — | — | — |

※ チェックポイントのダウンロード先: （準備中）

---

## ディレクトリ構成

```
.
├── scripts/              # 実行スクリプト（train / evaluate / prepare_dataset）
├── config/               # 実験設定ファイル
│   └── tsukuba_conf/     # 筑波大学データセット向け設定
├── easytsf/              # モデルライブラリ
│   ├── layer/            # KAN層・Transformer層・埋め込み層
│   └── model/            # モデル実装
├── tools/
│   ├── eval/             # 評価スクリプト
│   ├── plot/             # 可視化スクリプト
│   └── interpret/        # 解釈性解析スクリプト
├── data/                 # データ（NPYは除外・meta.jsonのみ管理）
├── checkpoints/          # チェックポイント（実体は外部管理）
├── figures/              # 論文掲載の代表図
└── archive/              # 旧実験コード（legacy ブランチ参照）
```

---

## 引用

```bibtex
@misc{mmkfusionpv2026,
  title  = {Multi-basis Mixture of KAN-experts for Solar PV Forecasting},
  author = {（著者名）},
  year   = {2026},
}
```

---

## ライセンス

MIT License — 詳細は [LICENSE](LICENSE) を参照してください。
