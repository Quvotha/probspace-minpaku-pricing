
from math import isclose, radians, atan, tan, sin, cos, sqrt, atan2, degrees, pi
from typing import Dict, List, Tuple, Iterable

import fasttext
from geopy.distance import geodesic
from joblib import delayed, Parallel
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
from transformers import BertTokenizer
import transformers

_BASE_DATE = '2015-05-01'


def _encode_1_host_id(host_df: pd.DataFrame, pairs: List[Tuple[str, str]],
                      continuous_features: List[str]) -> Dict[str, float]:
    assert host_df['host_id'].nunique() == 1
    encode = {'host_id': host_df['host_id'][0]}

    # `room_type`, `neighbourhood` で持ってる民泊の特徴を数値化する
    for n, r in pairs:
        encode[f'NR_{n}_{r}'] = host_df.query(f'neighbourhood == "{n}" and room_type == "{r}"').shape[0]

    # Frequency encoding
    encode['Freq_host_id'] = host_df.shape[0]

    # 数値項目の集計値を特徴として加える
    for c in continuous_features:
        encode[f'Min{c}'] = host_df[c].min()
        encode[f'Max{c}'] = host_df[c].max()
        encode[f'Range{c}'] = encode[f'Max{c}'] - encode[f'Min{c}']
        encode[f'Mean{c}'] = host_df[c].mean()
        encode[f'Median{c}'] = host_df[c].median()
        std = host_df[c].std()
        encode[f'Std{c}'] = 0. if np.isnan(std) else std
    return encode


class HostIDEncoder(BaseEstimator, TransformerMixin):
    """`host_id` をエンコードする。

    訓練データとテストデータで `host_id` の被りが無いことが前提のつくり。
    """

    def __init__(self, continuous_features: List[str]):
        """Initializer.

        Parameters
        ----------
        continuous_features : List[str]
            数値項目の名称をリストで指定する。
        """
        super(HostIDEncoder, self).__init__()
        self.continuous_features = continuous_features

    def fit(self, X: pd.DataFrame, y=None) -> object:
        """_summary

        Parameters
        ----------
        X : pd.DataFrame
            訓練データ。`room_type`, `neighbourhood` を含むこと。

        y :
            無視。

        Returns
        -------
        self
            訓練済みのエンコーダー。
        """
        self.pairs_ = [
            (n, r) for n in X['neighbourhood'].unique() for r in X['room_type'].unique().tolist()
            if isinstance(n, str) and isinstance(r, str)
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """`host_id` をエンコードする。

        Parameters
        ----------
        X : pd.DataFrame
            対象データ。
            `host_id`. `room_type`, `neighbourhood`, `continuous_features` で指定したカラム全てを含むこと。

        Returns
        -------
        encode : pd.DataFrame
            `host_id` をエンコードした結果。`host_id` 毎に1行。
        """
        assert X.shape[0] > 0

        encode = []
        for host_id, host_df in X.groupby('host_id'):
            encode_ = {'host_id': host_id}

            # `room_type`, `neighbourhood` で持ってる民泊の特徴を数値化する
            for n, r in self.pairs_:
                encode_[f'NR_{n}_{r}'] = host_df.query(f'neighbourhood == "{n}" and room_type == "{r}"').shape[0]

            # Frequency encoding
            encode_['Freq_host_id'] = host_df.shape[0]

            # 数値項目の集計値を特徴として加える
            for c in self.continuous_features:
                encode_[f'Min_{c}'] = host_df[c].min()
                encode_[f'Max_{c}'] = host_df[c].max()
                encode_[f'Range_{c}'] = encode_[f'Max{c}'] - encode_[f'Min{c}']
                encode_[f'Mean_{c}'] = host_df[c].mean()
                encode_[f'Median_{c}'] = host_df[c].median()
                std = host_df[c].std()
                encode_[f'Std_{c}'] = 0. if np.isnan(std) else std
            encode.append(encode_)
        encode = pd.DataFrame(encode)
        return encode


def _calculate_km(lat1: List[float],
                  longi1: List[float],
                  lat2: List[float],
                  longi2: List[float],
                  i1: int, i2: int) -> Tuple[int, int, float]:
    latitude1 = lat1[i1]
    longitude1 = longi1[i1]
    latitude2 = lat2[i2]
    longitude2 = longi2[i2]
    return i1, i2, geodesic((latitude1, longitude1), (latitude2, longitude2)).km


class NearestStations(BaseEstimator, TransformerMixin):

    def __init__(self, stations: pd.DataFrame, n_jobs: int = -1, verbose: int = 1):
        """Initializer.

        Parameters
        ----------
        stations : pd.DataFrame
            `station_name`, `latitude`, `longitude` を使用する。
            `station_name` に重複があると経度緯度を平均して重複を削除する。
        n_jobs, verbose : int
            `fit` で `joblib.Parallel` に渡す。
        """
        self.stations = stations.groupby('station_name')[['station_name', 'latitude', 'longitude']].mean().reset_index()
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y=None) -> object:
        """各民泊と駅の距離を計算する。

        Parameters
        ----------
        X : pd.DataFrame
            民泊の経度緯度情報。`latitude`, `longitude` を使用する。index を民泊のキーに使うので重複は認めない。
        y :
            無視。

        Returns
        -------
        self
        """
        assert not X.index.duplicated().any()
        # 民泊と駅の距離を計算する
        lat1 = X['latitude'].tolist()
        longi1 = X['longitude'].tolist()
        lat2 = self.stations['latitude'].tolist()
        longi2 = self.stations['longitude'].tolist()
        idx_pairs = [(i1, i2) for i1 in range(X.shape[0]) for i2 in range(self.stations.shape[0])]
        distances = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_calculate_km)(lat1, longi1, lat2, longi2, i1, i2) for i1, i2 in idx_pairs
        )

        # index * 駅名に pivot する
        data = []  # (index of `X`, statino_name, km)
        for i1, i2, km in distances:  # まずは (index, 駅名, km) に
            idx = X.index[i1]
            station_name = self.stations['station_name'].tolist()[i2]
            data.append([idx, station_name, km])
        del distances
        self.distances_ = pd.pivot_table(
            data=pd.DataFrame(data=data, columns=['index', 'station_name', 'km']),
            index='index',
            columns='station_name',
            values='km'
        )
        return self

    def kneighbors(self, k: int, idx) -> Tuple[List[str], List[float]]:
        """指定された民泊から近い駅の名前と距離を取得する。

        Parameters
        ----------
        k : int
            何駅分の名前と距離を取得するか。
        idx :
            民泊の識別子。存在しない値を指定すると KeyError.

        Returns
        -------
        nearest_stations, distances: Tuple[List[str], List[float]]
            民泊から近い `k` 個の駅情報。(駅名のリスト, 距離のリスト) 形式。

        Raises
        ------
        ValueError
            `k` が int ではない、又は 1 <= `k` <= `self.n_stations` を満たさない時。
        """
        if not (isinstance(k, int) and 1 <= k <= self.n_stations):
            raise ValueError(f'`k` MUST be interger, 1 <= `k` <= {self.n_stations}, but {k} was given.')
        distances = self.distances_.loc[idx].to_numpy()
        neighbor_indices = np.argsort(distances)[:k]
        dist_to_nearest_stations = distances[neighbor_indices]
        nearest_stations = self.distances_.columns[neighbor_indices].tolist()
        return nearest_stations, dist_to_nearest_stations

    @ property
    def n_stations(self) -> int:
        """駅の数
        """
        return self.stations.shape[0]

    @property
    def indices_(self) -> list:
        return self.distances_.index.tolist()


class NeighborMinpakuCounter(BaseEstimator, TransformerMixin):

    def __init__(self, n_jobs: int = -1, verbose: int = 1):
        """Initializer.

        Parameters
        ----------
        n_jobs, verbose : int
            `fit` で `joblib.Parallel` に渡す。
        """
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X: pd.DataFrame, y=None) -> object:
        """民泊間の距離を計算する。

        Parameters
        ----------
        X : pd.DataFrame
            民泊の経度緯度情報。`latitude`, `longitude` を使用する。index を民泊のキーに使うので重複は認めない。
        y :
            無視。

        Returns
        -------
        self
        """
        assert not X.index.duplicated().any()

        def _distances(latitude_values: Tuple[float],
                       longitude_values: Tuple[float],
                       i1: int) -> Tuple[int, List[float]]:
            lat1, longi1 = latitude_values[i1], longitude_values[i1]
            distances = [
                geodesic((lat1, longi1), (latitude_values[i2], longitude_values[i2])).km
                for i2 in range(len(latitude_values))
            ]
            return i1, distances

        latitude_values = tuple(X['latitude'].tolist())
        longitude_values = tuple(X['longitude'].tolist())

        indices_and_distances = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_distances)(latitude_values, longitude_values, i)
            for i in range(len(latitude_values))
        )
        data, indices = [], []
        for i, distances in indices_and_distances:
            indices.append(X.index[i])
            data.append(distances)
        del indices_and_distances
        self.distances_ = pd.DataFrame(data=data, index=indices, columns=indices)
        return self

        # latitude = X['latitude'].tolist()
        # longitude = X['longitude'].tolist()
        # idx_pairs = [(i1, i2) for i1 in range(X.shape[0]) for i2 in range(X.shape[0])]
        # distances = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
        #     delayed(_calculate_km)(latitude, longitude, latitude, longitude, i1, i2)
        #     for i1, i2 in idx_pairs
        # )
        # data = [(X.index[i1], X.index[i2], km) for i1, i2, km in distances]
        # del distances
        # distances = pd.DataFrame(data, columns=['from', 'to', 'km'])

        # distances = []
        # for idx1, lat1, longi1 in zip(X.index, X['latitude'].to_numpy(), X['longitude'].to_numpy()):
        #     for idx2, lat2, longi2 in zip(X.index, X['latitude'].to_numpy(), X['longitude'].to_numpy()):
        #         distances.append((idx1, idx2, geodesic((lat1, longi1), (lat2, longi2)).km))
        # distances = pd.DataFrame(data=distances, columns=['from', 'to', 'km'])
        # self.distances_ = pd.pivot_table(data=distances, index=['from'], columns=['to'], values=['km'])

    def count_neighbors(self, idx: int, km: float) -> int:
        """指定された民泊から一定距離内にいくつの民泊があるのかをカウントする。

        Parameters
        ----------
        idx: int
            民泊の識別子。存在しない値を指定すると KeyError.
        km : float
            距離の閾値(km).

        Returns
        -------
        count : int
            `row_id` で指定した民泊から `km` キロ以内にある民泊の数。自分自身は含まない。
        """
        mask = self.distances_.loc[idx] <= km
        count = mask.sum() - 1  # -1 は自分自身を除外するため
        return count

    @ property
    def indices_(self) -> list:
        return self.distances_.index.tolist()


def vincenty_inverse(lat1, lon1, lat2, lon2, default=np.nan):
    """2地点の経度緯度情報から距離と方位角を計算する。

    ほぼ以下のサイトのコードをコピペした。
    https://qiita.com/r-fuji/items/5eefb451cf7113f1e51b
    """

    # 楕円体
    ELLIPSOID_GRS80 = 1  # GRS80
    ELLIPSOID_WGS84 = 2  # WGS84

    # 楕円体ごとの長軸半径と扁平率
    GEODETIC_DATUM = {
        ELLIPSOID_GRS80: [
            6378137.0,         # [GRS80]長軸半径
            1 / 298.257222101,  # [GRS80]扁平率
        ],
        ELLIPSOID_WGS84: [
            6378137.0,         # [WGS84]長軸半径
            1 / 298.257223563,  # [WGS84]扁平率
        ],
    }

    # 反復計算の上限回数
    ITERATION_LIMIT = 1000

    # 差異が無ければ0.0を返す
    if isclose(lat1, lat2) and isclose(lon1, lon2):
        return 0.0

    # 計算時に必要な長軸半径(a)と扁平率(ƒ)
    # 楕円体はGRS80の値を用いる
    a, ƒ = 6378137.0, 1 / 298.257222101
    b = (1 - ƒ) * a

    φ1 = radians(lat1)
    φ2 = radians(lat2)
    λ1 = radians(lon1)
    λ2 = radians(lon2)

    # 更成緯度(補助球上の緯度)
    U1 = atan((1 - ƒ) * tan(φ1))
    U2 = atan((1 - ƒ) * tan(φ2))

    sinU1 = sin(U1)
    sinU2 = sin(U2)
    cosU1 = cos(U1)
    cosU2 = cos(U2)

    # 2点間の経度差
    L = λ2 - λ1

    # λをLで初期化
    λ = L

    # 以下の計算をλが収束するまで反復する
    # 地点によっては収束しないことがあり得るため、反復回数に上限を設ける
    for i in range(ITERATION_LIMIT):
        sinλ = sin(λ)
        cosλ = cos(λ)
        sinσ = sqrt((cosU2 * sinλ) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosλ) ** 2)
        cosσ = sinU1 * sinU2 + cosU1 * cosU2 * cosλ
        σ = atan2(sinσ, cosσ)
        sinα = cosU1 * cosU2 * sinλ / sinσ
        cos2α = 1 - sinα ** 2
        cos2σm = cosσ - 2 * sinU1 * sinU2 / cos2α
        C = ƒ / 16 * cos2α * (4 + ƒ * (4 - 3 * cos2α))
        λʹ = λ
        λ = L + (1 - C) * ƒ * sinα * (σ + C * sinσ * (cos2σm + C * cosσ * (-1 + 2 * cos2σm ** 2)))

        # 偏差が.000000000001以下ならbreak
        if abs(λ - λʹ) <= 1e-12:
            break
    else:
        # 計算が収束しなかった場合
        return default

    # λが所望の精度まで収束したら以下の計算を行う
    u2 = cos2α * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    Δσ = B * sinσ * (cos2σm + B / 4 * (cosσ * (-1 + 2 * cos2σm **
                                               2) - B / 6 * cos2σm * (-3 + 4 * sinσ ** 2) * (-3 + 4 * cos2σm ** 2)))

    # 2点間の楕円体上の距離
    s = b * A * (σ - Δσ)

    # 各点における方位角
    α1 = atan2(cosU2 * sinλ, cosU1 * sinU2 - sinU1 * cosU2 * cosλ)
    α2 = atan2(cosU1 * sinλ, -sinU1 * cosU2 + cosU1 * sinU2 * cosλ) + pi

    if α1 < 0:
        α1 = α1 + pi * 2
    return degrees(α1)


class LocationClustering(BaseEstimator, TransformerMixin):

    def __init__(self, kmeans_args: dict = {'n_clusters': 9, 'random_state': 901}):
        self.kmeans_args = kmeans_args

    def fit(self, X: pd.DataFrame, y=None) -> object:
        """位置情報のクラスタリングを行う。

        Parameters
        ----------
        X : pd.DataFrame
            経度緯度を `latitude`, `longitude` のカラムに設定しておくこと。
        y :
            無視する。

        Returns
        -------
        self : object
            Fit した結果。
        """
        scaler = StandardScaler().fit(X[['latitude', 'longitude']])
        X_scaled = scaler.transform(X[['latitude', 'longitude']])
        kmeans = KMeans(**self.kmeans_args).fit(X_scaled)
        self.scaler_ = scaler
        self.kmeans_ = kmeans
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Kmeans によるクラスタリング結果を用いた特徴量を返す。

        Parameters
        ----------
        X : pd.DataFrame
            民泊の経度緯度を `latitude`, `longitude` のカラムに設定しておくこと。

        Returns
        -------
        features : pd.DataFrame
            特徴量。次の2つのカラムを持つ。
            - `cluster_id`: 民泊がどのクラスターに属しているのかを示す
            - `km_from_center`: 民泊が自身が所属するクラスターのセントロイドから何 km 離れているかを示す
            - `azimuth`: クラスターの中心から民泊までの方位角
        """
        assert not X.index.duplicated().any()
        features = []
        # 所属するクラスターを得る
        X_scaled = self.scaler_.transform(X[['latitude', 'longitude']].copy())
        cluster_ids = self.kmeans_.predict(X_scaled)
        # クラスターの中心からの距離を得る
        centroids = self.scaler_.inverse_transform(self.kmeans_.cluster_centers_, copy=True)
        for cluster_id, latitude, longitude in zip(cluster_ids, X['latitude'].to_numpy(), X['longitude'].to_numpy()):
            centroid = centroids[cluster_id]
            latitude_center, longitude_center = centroid
            distance = geodesic((latitude, longitude), (latitude_center, longitude_center)).km
            azimuth = vincenty_inverse(latitude_center, longitude_center, latitude, longitude, 2)
            features.append([cluster_id, distance, azimuth])
        features = pd.DataFrame(data=features, index=X.index, columns=['cluster_id', 'km_from_center', 'azimuth'])
        return features


class BertSequenceVectorizer:

    def __init__(self, model_name: str, max_len: int):
        """Initializer.

        このクラスは以下の記事で紹介されているものを微改修したもの。
        https://www.guruguru.science/competitions/16/discussions/fb792c87-6bad-445d-aa34-b4118fc378c1/

        `model_name`, `max_len` をパラメータ化した。
        Bert model を明示的に evaluation mode にするようにした。`vectorize` は torch.inference_mode で実行するようにした。
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.bert_model.eval()
        self.max_len = max_len

    @ torch.inference_mode()
    def vectorize(self, sentence: str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        bert_out = self.bert_model(inputs_tensor, masks_tensor)
        seq_out, pooled_out = bert_out['last_hidden_state'], bert_out['pooler_output']
        if torch.cuda.is_available():
            return seq_out[0][0].cpu().detach().numpy()  # 0番目は [CLS] token, 768 dim の文章特徴量
        else:
            return seq_out[0][0].detach().numpy()


class LanguageWiseBertVectorizer(BaseEstimator, TransformerMixin):

    LANGUAGE_LABELS = {
        'english': '__label__en',
        'japanese': '__label__ja',
        'chinese': '__label__zh',
    }

    def __init__(
            self, model_name_en: str, model_name_ja: str, model_name_zh: str, language_detection_model_path: str,
            max_len: int = 128):
        """Initializer.

        言語識別モデルで判定した言語によって適切な BERT pretrained model を使いベクトル化する。
        英語、日本語、中国語のみ識別し、他は一律で英語扱いする。

        Parameters
        ----------
        model_name_en, model_name_ja, model_name_zh : str
            それぞれ英語、日本語、中国語の BERT pretrained model の名称。
            Huggingface の repository からダウンロードしてくるので repository に存在するモデルを指定すること。
        fattext_model_path : str
            言語識別モデルのパス。
            以下のサイトから『lid.176.bin』をダウンロードして使用するのでそのパスを指定する。
            https://fasttext.cc/docs/en/language-identification.html
            これらのモデルは Refferences [1], [2] の文献を参考に開発されたもの。
        max_len : int, optional
            ベクトル化する文章長の最大値。

        Refferences
        -----------
        [1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, Bag of Tricks for Efficient Text Classification
        [2] A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, FastText.zip: Compressing text classification models
        """
        self.language_detection_model = fasttext.load_model(language_detection_model_path)
        self.bert_model_en = BertSequenceVectorizer(model_name_en, max_len)
        self.bert_model_ja = BertSequenceVectorizer(model_name_ja, max_len)
        self.bert_model_zh = BertSequenceVectorizer(model_name_zh, max_len)

        self.model_name_en = model_name_en
        self.model_name_ja = model_name_ja
        self.model_name_zh = model_name_zh
        self.language_detection_model_path = language_detection_model_path
        self.max_len = max_len

    def fit(self, X=None, y=None) -> object:
        """何もしない"""
        return self

    def transform(self, X: Iterable[str]) -> pd.DataFrame:
        """文章の埋め込みを Bert model によって得る。

        Parameters
        ----------
        X : Iterable[str]
            埋め込みを得たい文章を iterable で渡す。

        Returns
        -------
        embedding_label_probability : pd.DataFrame, shape = (X.shape[0]. 770)
            Bert model で得た768次元の埋め込みに language detection model で得た言語のラベルと確率をくっつけたもの。
        """
        assert not X.index.duplicated().any()
        # 埋め込み、言語を示すラベル、確率
        embedding, labels, probabilities = [], [], []
        for sentence in X:
            # 言語識別モデルで各文章が何語かを予測する
            # 予測した言語に応じて埋め込みの取得に使う Bert model を買える
            prediction = self.language_detection_model.predict(sentence)
            probability = prediction[1][0]
            probabilities.append(probability)
            label = prediction[0][0]
            labels.append(label)
            if label == self.LANGUAGE_LABELS['english']:
                emb = self.bert_model_en.vectorize(sentence)
            elif label == self.LANGUAGE_LABELS['japanese']:
                emb = self.bert_model_ja.vectorize(sentence)
            elif label == self.LANGUAGE_LABELS['chinese']:
                emb = self.bert_model_zh.vectorize(sentence)
            else:
                emb = self.bert_model_en.vectorize(sentence)
            embedding.append(emb)
        embedding = pd.DataFrame(embedding, index=X.index)
        n_dim = embedding.shape[1]
        embedding.columns = [f'embedding{i + 1}' for i in range(n_dim)]
        labels = pd.DataFrame(labels, index=X.index, columns=['language_label'])
        probabilities = pd.DataFrame(probabilities, index=X.index, columns=['language_probability'])
        return pd.concat([labels, probabilities, embedding], axis=1)
