# archive/

このフォルダには、実験過程で使用した旧スクリプト・中間結果・試行錯誤のコードが保存されています。

## 概要

| サブフォルダ | 内容 |
|---|---|
| `old_training_scripts/` | 旧版学習スクリプト（`traingraph.py`, `traingraph_V2.py` 等） |
| `old_models/` | 旧版モデルコード（`MMK_Mix_RevIN`, `TimeLLM` 等） |
| `old_configs/` | 未使用・実験中の設定ファイル群 |
| `old_results/` | 実験中間結果の画像 |
| `old_save/` | 旧チェックポイント |
| `old_analysis/` | 解析・可視化スクリプト（非再現系） |
| `old_datasets/` | 前処理過程のデータ（prepared以外） |
| `misc/` | 日付別結果フォルダ等 |

## 注意

- このフォルダは `.gitignore` に含まれていません（`main` ブランチに履歴を残すため）
- ただし、大容量ファイル（`.npy`, `.ckpt` 等）は `.gitignore` で除外されます
- 本番環境での再現には `main` ブランチのルート構成のみを使用してください
- 実験の詳細な経緯は `legacy` ブランチに保存されています

## legacy ブランチへの切り替え

```bash
git checkout legacy
```
