# probspace-minpaku-pricing

Probspace で開催された [民泊サービスの宿泊料金予測](https://comp.probspace.com/competitions/bnb_price) での作成物を保管しています。

## 順位

||RMSLE|Rank|
|:---|---:|---:|
|Public LB|0.80580|39|
|Private LB|0.74878|34|

## 特徴量エンジニアリング

- `name`, `neighbourhood`, `latitude`, `longitude`, `room_type`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `availability_365` を特徴量に使用
  - `reviews_per_month` の欠損値は -1 で補完
  - `name` は表記ゆれを正規化
- テキトーな固定日付から `last_review` までの経過日数
- `name` は次の手順でベクトル化
  - 言語ごとに異なる形態素解析器を用いて分かち書き形式に変換
    - 言語予測には Fasttext を使用
    - 日本語は sudachipy 中国語は jieba で分かち書き形式に変換
    - 日本語と中国語以外は分かち書き形式に変換せずそのまま使用
  - TF-IDFを得る
  - TruncatedSVD で64次元に圧縮
  - NMF で64次元に圧縮
- 民泊を座標(`latitude`, `longitude`)でクラスタリングし所属クラスター・セントロイドからの距離・セントロイドからの方位角を計算
  - クラスタリングは `KMeans(n_clusters=9)` で行う

## モデル

CatBoostRegressor を用いた。`y` は `np.log1p` で対数変換して学習させる。最小化の目標関数は RMSE とし、その他のハイパラは `cat_features`, `random_state` を指定した。

## 交差検証

`neighbourhood` で Stratified K-Fold(n_splits=5). 全ての Fold の予測結果の算術平均値を submit した。

## コード
[このNotebook](/notebooks/066.ipynb).

## ポエム

Public LB 1桁の方々が公開しているソリューションを見たところ、私とは比較にならないくらい `name` の処理に凝っているなとの所感です。NLPのコンペだったと言う方すら居たのが印象的でした。データを見る力が足りず、私はそこまでできませんでした。

`name` については『特徴量エンジニアリング』で述べたものの他にも Bert の Pretrained model を使った embedding も試しました。効果はありましたが、古典的な TF-IDF -> 次元削減で得た特徴を用いた方がスコアが良く、両者を同時に用いるとTF-IDF -> 次元削減を単体で用いた場合よりもスコアが悪化しました。

民泊の位置情報を用いて『自身から一定距離にある他の民泊数』『近い最寄り駅TOP N個 + それらまでの距離』等の特徴量も試しましたが、スコアは伸びるものの、クラスタリングを用いた特徴の方がスコアが改善しました。これらの特徴量を同時に使用するとクラスタリングの特徴量を単体で用いた場合よりもスコアが悪化しました。

`host_id` を用いたエンコーディングは Leakage の元でした。同じような民泊物件でもホストによる根付けのクセの差が大きかったのかなと推察しています。

交差検証は `y` を binning したものでの Stratified K-Fold, `host_id` による Group K-Fold も試しました。おおよそ Local CV と Public LB は相関していましたが両者の間で順位の不一致があったりして最後まで悩んだあげく、最後はテキトーに決めました。