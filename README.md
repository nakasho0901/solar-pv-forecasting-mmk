# MMK-FusionPV：太陽光発電量の96時間予測と内部関数の分析

過去96時間の発電量・気象・時刻情報から、続く96時間の発電電力量を予測する学部研究です。既存の **MMK（Multi-layer Mixture-of-KAN network）** を太陽光発電量予測に適用し、iTransformerとの精度比較、expertの混合重みの可視化、内部基底関数の数式抽出に取り組みました。

このREADMEでは、最終構成として整理した **MMK_FusionPV_FeatureToken_v2（A11・stride1）** と、その比較結果を扱います。

## 研究の目的

予測誤差の大小に加え、モデル内部でどの種類の関数が利用されているかを調べることを目的としました。

複数種類のKANを混合するMMKを利用し、太陽光発電量の周期性を取り入れた予測と、学習された内部表現の分析を行います。MMKの基本手法は既存研究に基づき、本研究ではPV予測への適用・構成変更・評価・分析を扱っています。

## 最終モデルの構成

- モデル：[MMK_FusionPV_FeatureToken_v2.py](easytsf/model/MMK_FusionPV_FeatureToken_v2.py)
- 設定：[MMK_FusionPV_FeatureToken_A11_stride1_v2.py](config/tsukuba_conf/MMK_FusionPV_FeatureToken_A11_stride1_v2.py)

| 処理 | 内容 |
|---|---|
| 入力 | 過去96時間 × 12特徴量 |
| PV正規化 | 入力窓の平均・標準偏差を使うPV専用RevIN |
| 特徴量のトークン化 | 各特徴量の96時点を共有のLinear層等で64次元に変換 |
| expert混合 | 3層。特徴量トークンごとに4expertをsoftmaxで重み付け |
| expertの種類 | Spline/KAN、Wavelet、Taylor、Jacobi |
| トークン集約 | 特徴量方向の平均 |
| 予測ヘッド | MLPによる96時点の補正値の出力 |
| 基準予測 | 過去の末尾24時間を繰り返した系列 |
| 最終出力 | 基準予測＋補正値を逆正規化し、非負制約を適用 |

4expertをすべて計算して混合する構造です。Taylor expertは定数項から3次項までの多項式を使います。

ゲートの形状は各層で「窓 × 特徴量 × expert」です。入力特徴量トークンごとに混合比率を求めるもので、96個の予測時刻それぞれにゲートを割り当てる構造ではありません。

## 本研究で扱った実装・分析

### 特徴量ごとの関数の使い分け

発電量・気象・時刻・ラグ情報を特徴量ごとのトークンに変換し、それぞれに適したexpertの混合比率を学習する構成を用いました。層ごとにゲートを取得できるため、学習後に混合比率を調べられます。

### 発電量の周期性を利用した予測

PV専用のRevINと、直近24時間の繰り返しを基準とする補正予測を組み込みました。出力には非負制約を適用しています。

これらは公開モデルで確認できる構成です。各要素を単独で除去した比較による精度向上の切り分けは、ここでは主張していません。

### 平均誤差と内部表現の両面からの評価

iTransformerとの予測精度の比較に加え、expert混合重みの可視化とTaylor基底関数の抽出を行いました。予測精度だけで優劣を判断せず、内部で学習された関数も分析対象としています。

## データと予測条件

| 項目 | 内容 |
|---|---|
| データ間隔 | 1時間 |
| 入力期間 | 過去96時間 |
| 予測期間 | 直後から96時間の各1時間 |
| 入力特徴量数 | 12 |
| stride | 1時間 |
| 学習窓数 | 19,726 |
| 検証窓数 | 2,818 |
| テスト窓数 | 5,636 |
| 予測対象 | 1時間ごとのPV発電電力量（kWh） |

窓数は [公開データ仕様](data/prepared/96-96_fusionA11_96_96_stride1/README.md) に記載された値です。窓は時間方向に重なるため、窓数を独立した日数・独立観測数とは扱いません。

### データの出典

過去の研究記録で使用元として挙げられている公開データは、Vinkらの *Multiyear microgrid data from a research building in Tsukuba, Japan* です。NIMS（物質・材料研究機構）の研究棟で計測されたデータで、原PV計測の対象期間は2015年1月1日～2018年4月24日です。

原データの期間と、気象結合後に本研究で実際に使用した期間は区別しています。最終データの開始・終了日および学習・検証・テストそれぞれのカレンダー期間は、最終meta.json・時刻配列との照合前です。

- [データ論文](https://doi.org/10.1038/sdata.2019.20)
- [Figshare collection](https://doi.org/10.6084/m9.figshare.c.4176698)

### 入力特徴量

| 分類 | 特徴量 |
|---|---|
| 発電量 | pv_kwh |
| 気象 | temp_c、rh_pct、ghi_wm2 |
| 時刻・昼夜 | hour_sin、hour_cos、is_daylight |
| 発電量ラグ | lag_pv_kwh_24h |
| 日射量ラグ | lag_ghi_wm2_1h、lag_ghi_wm2_2h、lag_ghi_wm2_3h、lag_ghi_wm2_24h |

## 予測結果

研究記録で最終結果として整理されている比較値は以下のとおりです。評価条件は「テスト5,636窓・96時間予測・夜間ゼロ補正（night0）」として記録されています。

| モデル | MAE（kWh） | RMSE（kWh） | パラメータ数 |
|---|---:|---:|---:|
| MMK-FusionPV | 8.6727 | 18.4701 | 約399K |
| iTransformer | 8.6386 | 17.3306 | 約1.8M |

MMK-FusionPVは、記録上のパラメータ数がiTransformerの約22%で、MAEは近い値となりました。一方、MAE・RMSEはいずれもiTransformerの方が小さく、MMK-FusionPVが予測精度で上回った結果ではありません。

パラメータ数の違いから、実測の推論速度や使用メモリの差まで結論付けることはできません。また、複数seedによる統計的な同等性は未確認です。

**数値の確認状況：** 上表は過去の研究記録に基づきます。卒業論文の最終表、元の評価ログ、両モデルのcheckpointとの直接照合は未完了です。MSE・nRMSE・Day-MAEは最終値を確認できていないため掲載していません。

### 評価条件の注意

night0は夜間の系列を0にする補正であり、昼間の時点だけを抽出する評価とは異なります。マスクの定義、実測系列への適用、誤差の平均対象について、最終評価コードとの照合が必要です。

時刻別・予測日別の平均波形は、個々の実在する4日間の予測とは異なる集計です。平均波形の一致だけから、各日のピーク予測が正確だとは判断していません。

## 内部表現の分析

### 層ごとのexpert混合重み

[保存されたversion12のゲート集計](archive/misc/2002_misc/MASTER_SUMMARY_v12_FULL.csv) では、表示された6特徴量について、次の傾向が確認できます。

| 層（0始まり） | 保存された集計の傾向 |
|---|---|
| Layer0 | TaylorとJacobiへの配分が大きい |
| Layer1 | 特徴量によってTaylorまたはJacobiへの配分が大きい |
| Layer2 | Taylorへの配分が約0.76～0.86 |

これは、対象トークンでexpertを混合する際の重みです。元特徴量が最終予測を何%決めたかを表す指標ではありません。また、混合重みが小さいexpertを除去しても性能を維持できることは検証していません。

### Taylor基底関数の数式抽出

TaylorKANの学習済み係数から、内部の基底関数を取り出します。

$$
\phi_{o,i}(z_i)=\sum_{k=0}^{3}c_{o,i,k}z_i^k
$$

ここで、z_iは内部トークンの成分です。元の気温・日射量や、最終的な発電量そのものを直接表す変数ではありません。

[内部基底の分析コード](tools/interpret/rank_taylor_layer2_contrib.py) では、各基底関数の平均絶対出力を使ってランキングします。

$$
C_{o,i}=\operatorname{mean}\left|\phi_{o,i}(z_i)\right|
$$

この値は対象層のTaylor expert内部での出力の大きさを示します。ゲート、残差、他expert、予測ヘッドまで含めた最終予測への寄与率ではありません。

本研究で示す解釈性は、**expertの混合比率と、内部で学習された関数を調べられること**です。最終予測の完全な説明や、太陽光発電の物理法則の抽出を達成したとは主張していません。

## 対象ファイル

| 内容 | ファイル |
|---|---|
| 最終モデルの構成 | [MMK_FusionPV_FeatureToken_v2.py](easytsf/model/MMK_FusionPV_FeatureToken_v2.py) |
| 最終構成の設定 | [MMK_FusionPV_FeatureToken_A11_stride1_v2.py](config/tsukuba_conf/MMK_FusionPV_FeatureToken_A11_stride1_v2.py) |
| KAN層の実装 | [kanlayer.py](easytsf/layer/kanlayer.py) |
| Taylor係数の抽出 | [extract_taylorkan_formulas.py](tools/interpret/extract_taylorkan_formulas.py) |
| 内部基底のランキング | [rank_taylor_layer2_contrib.py](tools/interpret/rank_taylor_layer2_contrib.py) |
| 保存済みゲート集計 | [MASTER_SUMMARY_v12_FULL.csv](archive/misc/2002_misc/MASTER_SUMMARY_v12_FULL.csv) |

## 再現性・確認状況

前処理済みデータ、最終meta.json、学習済み重み、最終評価ログは現在の公開構成に含まれていません。公開スクリプトにはモデル分岐やパスの不整合もあるため、最終結果を再現する一連の実行手順は検証済みのものとして提示していません。

最終checkpointで実際に使用した損失関数は未確認のため、特定の損失を最終採用として記載していません。損失変更によるピーク誤差の改善も、同条件の前後比較を確認できるまで成果として断定しません。

各splitの対象時刻の重複、気象データの取得・結合条件、昼夜マスクの定義も、最終ファイルとの照合が残っています。

## 既存手法と生成AIの利用

MMK、KAN、iTransformer、RevINなどの既存手法・公開実装を利用しています。研究過程では、コード作成・修正の支援に生成AIも使用しました。

本研究の実装・分析対象は上記に示したとおりです。基本手法の提案と本研究での応用・改良を区別し、リポジトリ全体を独自のフルスクラッチ実装とは位置付けていません。

## 参考文献

- Xiao Han, Xinfeng Zhang, Yiling Wu, Zhenduo Zhang, Zhe Wu. *Are KANs Effective for Multivariate Time Series Forecasting?* arXiv:2408.11306, 2024 / revised 2025. [論文](https://arxiv.org/abs/2408.11306)
- Karina Vink, Eriko Ankyu, Michihisa Koyama. *Multiyear microgrid data from a research building in Tsukuba, Japan.* Scientific Data 6, 190020, 2019. [論文](https://doi.org/10.1038/sdata.2019.20)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)

## ライセンス

[MIT License](LICENSE)。第三者のコード・データには、それぞれの出典と利用条件が適用されます。
