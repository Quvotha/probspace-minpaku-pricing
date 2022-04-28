from typing import Dict, List, Tuple

from geopy.distance import geodesic
from joblib import delayed, Parallel
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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


class NearestStationsFinder():

    def __init__(self, locations: pd.DataFrame, stations: pd.DataFrame, reduce: bool = False):
        """Initializer.

        Parameters
        ----------
        locations : pd.DataFrame
            民泊の経度緯度情報。`row_id`, `latitude`, `longitude` を使用する。
            `row_id` は識別子なので欠損や重複は認められない。
        stations : pd.DataFrame
            `station_name`, `latitude`, `longitude` を使用する。
        reduce : bool, optional
            True なら `stations` の駅名の重複を削除する。経度緯度は平均値が用いられる。
        """
        assert not locations['row_id'].duplicated().any()
        assert not locations['row_id'].isnull().any()
        self.locations = locations[['row_id', 'latitude', 'longitude']]
        if reduce:
            stations = stations.groupby('station_name').mean().reset_index()
        self.stations = stations[['station_name', 'latitude', 'longitude']]
        self._calculate()

    def _calculate(self) -> None:
        """各民泊と駅の距離を計算する。
        """
        distances = []
        for lat1, longi1 in zip(self.locations['latitude'].to_numpy(), self.locations['latitude'].to_numpy()):
            distance = [
                geodesic((lat1, longi1), (lat2, longi2)).km
                for lat2, longi2 in zip(self.stations['latitude'].to_numpy(), self.stations['latitude'].to_numpy())
            ]
            distances.append(distance)
        self.distances = pd.DataFrame(distances, columns=self.stations['station_name'], index=self.locations['row_id'])

    def get_nearest_stations(self, k: int, row_id) -> Tuple[List[str], List[float]]:
        """指定された民泊から近い駅の名前と距離を取得する。

        Parameters
        ----------
        k : int
            何駅分の名前と距離を取得するか。
        row_id : _type_
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
        distances = self.distances.loc[row_id].to_numpy()
        neighbor_indices = np.argsort(distances)[:k]
        dist_to_nearest_stations = distances[neighbor_indices]
        nearest_stations = self.distances.columns[neighbor_indices].tolist()
        return nearest_stations, dist_to_nearest_stations

    @ property
    def n_stations(self) -> int:
        """駅の数
        """
        return self.stations.shape[0]
