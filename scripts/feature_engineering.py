from math import isclose, radians, atan, tan, sin, cos, sqrt, atan2, degrees, pi
from typing import Dict, List, Tuple

from geopy.distance import geodesic
from joblib import delayed, Parallel
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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


class NeighborMinpakuCounter:

    def __init__(self, locations: pd.DataFrame):
        """Initializer.

        Parameters
        ----------
        locations : pd.DataFrame
            民泊の経度緯度。`row_id`, `latitude`, `longitude` を使用する。
            `row_id` は int, 識別子として使うので欠損や重複は認められない。
        """
        assert not locations['row_id'].duplicated().any()
        assert not locations['row_id'].isnull().any()
        self.locations = locations[['row_id', 'latitude', 'longitude']]

    def calculate_distance(self, verbose: int = 1, n_jobs: int = -1) -> None:
        """民泊間の距離を計算する。

        Parameters
        ----------
        verbose, n_jobs : int, optional
            joblob.Parallel に渡す。
        """
        def _calculate(id_and_location1, id_and_location2):
            row_id1, latitude1, longitude1 = id_and_location1
            row_id2, latitude2, longitude2 = id_and_location2
            distance = geodesic((latitude1, longitude1), (latitude2, longitude2)).km
            return (int(row_id1), int(row_id2), distance)

        locations = self.locations.to_numpy()
        distances = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(_calculate)(loc1, loc2) for loc1 in locations for loc2 in locations
        )
        # distances = []
        # for loc1 in locations:
        #     for loc2 in locations:
        #         row_id1, lat1, longi1 = loc1
        #         row_id2, lat2, longi2 = loc2
        #         distance = geodesic((lat1, longi1), (lat2, longi2)).km
        #         distances.append((row_id1, row_id2, distance))
        distances = pd.DataFrame(distances, columns=['from', 'to', 'km'])
        self.distances = pd.pivot_table(data=distances, index=['from'], columns=['to'], values=['km'])

    def count_minpaku(self, row_id: int, km: float) -> int:
        """指定された民泊から一定距離内にいくつの民泊があるのかをカウントする。

        Parameters
        ----------
        row_id : int
            民泊を識別するID.
        km : float
            距離の閾値(km).

        Returns
        -------
        count : int
            `row_id` で指定した民泊から `km` キロ以内にある民泊の数。自分自身は含まない。
        """
        if self.distances is None:
            raise ValueError('`calculate_distance` must be called before.')
        mask = self.distances.loc[row_id] <= km
        count = mask.sum() - 1  # -1 は自分自身を除外するため
        return count


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
