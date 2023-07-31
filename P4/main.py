import pandas as pd
import numpy as np
from functools import wraps
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self


_CLEANUP = False


def array_to_str(arr: np.ndarray) -> str:
    if arr.ndim == 1:
        return ','.join(map(str, np.around(arr, 4).tolist()))
    elif arr.ndim == 2:
        return '\n'.join((','.join(map(str, x)) for x in np.around(arr, 4).tolist()))
    else:
        raise ValueError('Only 1 and 2-d arrays suppported')


def rescale(x: np.ndarray) -> np.ndarray:
    min_: float = np.min(x)
    range_: float = np.ptp(x)
    return (x - min_) / range_


class HierarchicalClustering:
    class Clustering:
        @staticmethod
        def _compute_distances(X: np.ndarray) -> np.ndarray:
            n = X.shape[0]
            distance_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(i + 1, n):
                    distance_matrix[i, j] = np.linalg.norm(X[i] - X[j])

            distance_matrix += distance_matrix.T

            return distance_matrix

        @staticmethod
        def _single(distance_matrix: np.ndarray, num_clusters: int) -> tuple:
            num_points = distance_matrix.shape[0]
            clusters = [[i] for i in range(num_points)]

            while len(clusters) > num_clusters:
                min_distance = np.inf
                merge_index1, merge_index2 = None, None

                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        cluster1 = clusters[i]
                        cluster2 = clusters[j]

                        distance: float = np.min(
                            distance_matrix[np.ix_(cluster1, cluster2)])

                        if distance < min_distance:
                            min_distance = distance
                            merge_index1, merge_index2 = i, j

                # Merge the two closest clusters
                assert merge_index1 is not None and merge_index2 is not None
                clusters[merge_index1].extend(clusters[merge_index2])
                del clusters[merge_index2]

                combined_cluster = clusters[merge_index1]
                for i in range(distance_matrix.shape[0]):
                    if i in combined_cluster:
                        continue
                    distance = np.min(
                        distance_matrix[np.ix_(combined_cluster, [i])])
                    distance_matrix[i, combined_cluster] = distance
                    distance_matrix[combined_cluster, i] = distance

            return tuple(map(tuple, clusters))

        @staticmethod
        def _complete(distance_matrix: np.ndarray, n_clusters: int) -> tuple:
            num_points = distance_matrix.shape[0]
            clusters = [[i] for i in range(num_points)]

            while len(clusters) > n_clusters:
                min_max_distance = np.inf
                merge_index1, merge_index2 = None, None

                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        cluster1 = clusters[i]
                        cluster2 = clusters[j]

                        max_distance: float = np.max(
                            distance_matrix[np.ix_(cluster1, cluster2)])

                        if max_distance < min_max_distance:
                            min_max_distance = max_distance
                            merge_index1, merge_index2 = i, j

                assert merge_index1 is not None and merge_index2 is not None
                clusters[merge_index1].extend(clusters[merge_index2])
                del clusters[merge_index2]

                combined_cluster = clusters[merge_index1]
                for i in range(distance_matrix.shape[0]):
                    if i in combined_cluster:
                        continue
                    distance: float = np.max(
                        distance_matrix[np.ix_(combined_cluster, [i])])
                    distance_matrix[i, combined_cluster] = distance
                    distance_matrix[combined_cluster, i] = distance

            return tuple(map(tuple, clusters))

    _METHODS = {
        'single': Clustering._single,
        'complete': Clustering._complete,
    }

    def __init__(self, n_clusters: int = 2, linkage: str = 'single'):
        self._n_clusters = n_clusters

        assert HierarchicalClustering._is_linkage_valid(linkage), (
            f'Invalid Linkage, valid linkages are \'single\' and '
            f'\'complete\', found {linkage!r}'
        )

        self._linkage = linkage

    def _build(self, X: np.ndarray):
        distance_matrix = HierarchicalClustering.Clustering._compute_distances(
            X)
        cluster_func = HierarchicalClustering._METHODS.get(self._linkage)
        if cluster_func is None:
            raise ValueError(
                f'Invalid Linkage, the Hierarchical Clustering instance '
                f'may have been tampered with. found {self._linkage!r}'
            )

        self._clusters = cluster_func(distance_matrix, self._n_clusters)

    def fit(self, X: np.ndarray) -> Self:
        self._build(X)
        return self

    def predict(self, X: np.ndarray):
        pass

    @property
    def clusters(self):
        return self._clusters

    @clusters.setter
    def clusters(self, value):
        raise ValueError(
            'The `clusters` attribute should not be modified externally!')

    @staticmethod
    def _is_linkage_valid(linkage: str) -> bool:
        return linkage in HierarchicalClustering._METHODS.keys()


class KMeans:
    class Clustering:
        @staticmethod
        def assign_points(centroids: np.ndarray, X: np.ndarray) -> np.ndarray:
            distances = KMeans.Clustering.compute_distances(centroids, X)

            return np.argmin(distances, axis=1)

        @staticmethod
        def choose_centroids(X: np.ndarray, k: int = 2) -> np.ndarray:
            return np.random.default_rng().choice(
                X,
                size=k,
                replace=False,
                shuffle=False
            )

        @staticmethod
        def compute_centroids(X: np.ndarray, assignments: np.ndarray, k: int) -> np.ndarray:
            centroids = np.empty((k, X.shape[1]))
            for i in range(k):
                idxs = assignments == i
                centroids[i] = np.mean(X[idxs], axis=0)
            return centroids

        @staticmethod
        def compute_distances(centroids: np.ndarray, X: np.ndarray) -> np.ndarray:
            return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        @staticmethod
        def compute_distortion(
                centroids: np.ndarray,
                X: np.ndarray,
                assignments: np.ndarray) -> float:
            _distortion = 0.

            for i in range(centroids.shape[0]):
                idxs = assignments == i
                _distortion += np.sum((X[idxs] - centroids[i]) ** 2)

            return _distortion

    def __init__(self, k: int = 2, max_iters: int = 100, tol: float = 1e-5, ):
        self._k = k
        self._max_iters = max_iters
        self._tol = abs(tol)

    def _build(self, X: np.ndarray):
        self._centroids = KMeans.Clustering.choose_centroids(X, self.k)

        old_centroids = np.empty_like(self._centroids)
        for _ in range(self._max_iters):
            np.copyto(old_centroids, self._centroids)

            assignments = self.predict(X)

            self._centroids = KMeans.Clustering.compute_centroids(
                X, assignments, self.k)

            if np.all(np.abs(self.centroids - old_centroids) < self._tol):
                self._n_iters = _
                break

        self._labels = self.predict(X)
        self._distortion = KMeans.Clustering.compute_distortion(
            self._centroids, X, self._labels)

    def fit(self, X: np.ndarray) -> Self:
        self._build(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return KMeans.Clustering.assign_points(self._centroids, X)

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self):
        raise ValueError(
            'The `labels` attribute should not be modified externally!'
        )

    @property
    def distortion(self):
        return self._distortion

    @distortion.setter
    def distortion(self):
        raise ValueError(
            'The `distortion` attribute should not be modified externally!'
        )

    @property
    def centroids(self):
        return self._centroids

    @centroids.setter
    def centroids(self):
        raise ValueError(
            'The `centroids` attribute should not be modified externally!'
        )

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self):
        raise ValueError(
            'The `k` attribute should not be modified externally!'
        )

    @property
    def n_iters(self):
        return self._n_iters

    @n_iters.setter
    def n_iters(self, value):
        raise ValueError(
            'The `n_iters` attribute should not be modified externally!'
        )


def _():
    pass


FuncType = type(_)
del _


def save_result(func_or_filename):
    if s := isinstance(func_or_filename, str):
        file = func_or_filename
    elif isinstance(func_or_filename, FuncType):
        file = f'{func_or_filename.__name__}.txt'
    else:
        raise TypeError(f'save_result only accepts types {str} and {FuncType}')

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            result = func(*args, **kwargs)
            if isinstance(result, np.ndarray):
                result_ = array_to_str(result)
            else:
                result_ = str(result)
            with open(file, 'w') as f:
                f.write(result_)
            return result
        return inner

    return decorator if s else decorator(func_or_filename)


@save_result
def q1(cum_data: np.ndarray) -> np.ndarray:
    return cum_data[(4, 48), :]  # 4 -> California, 48 -> Wisconsin


@save_result
def q2(data: np.ndarray) -> np.ndarray:
    return data[(4, 48), :]  # 4 -> California, 48 -> Wisconsin


@save_result
def q3() -> str:
    return '\n'.join((
        'Mean of the time-differenced data',
        'Standard deviation of the time-differenced data',
        'Median of the time-differenced data',
        'Linear trend coefficient of the data',
        'Auto-correlation of the data',
    ))


@save_result
def q4(data: np.ndarray) -> np.ndarray:
    mean = np.mean(data, axis=1)
    mean = rescale(mean)

    std = np.std(data, axis=1)
    std = rescale(std)

    median = np.median(data, axis=1)
    median = rescale(median)

    half_length = (data.shape[-1] + 1) / 2
    ltc = np.array([sum((x[i] - mean[idx]) * (i + 1 - half_length) for i in range(len(x)))
                    for idx, x in enumerate(data.tolist())]) / sum((i + 1 - half_length) ** 2 for i in range(data.shape[-1]))
    ltc = rescale(ltc)

    ac = np.array([sum(
        (x[i] * x[i - 1] for i in range(1, len(x)))
    ) / np.sum(np.power(x, 2)) for x in (data - mean.reshape(50, -1)).tolist()])
    ac = rescale(ac)

    parametric_data = np.empty((50, 5))
    parametric_data[:, 0] = mean
    parametric_data[:, 1] = std
    parametric_data[:, 2] = median
    parametric_data[:, 3] = ltc
    parametric_data[:, 4] = ac

    return parametric_data


@save_result
def q5(parametric_data: np.ndarray) -> np.ndarray:
    slhc = HierarchicalClustering(n_clusters=5)
    slhc = slhc.fit(parametric_data)

    idxs = np.arange(parametric_data.shape[0])

    for cluster_idx, cluster in enumerate(slhc.clusters):
        for item in cluster:
            idxs[item] = cluster_idx

    return idxs


@save_result
def q6(parametric_data: np.ndarray) -> np.ndarray:
    clhc = HierarchicalClustering(n_clusters=5, linkage='complete')
    clhc = clhc.fit(parametric_data)

    idxs = np.arange(parametric_data.shape[0])

    for cluster_idx, cluster in enumerate(clhc.clusters):
        for item in cluster:
            idxs[item] = cluster_idx

    return idxs


@save_result
def q7(kmeans: KMeans) -> np.ndarray:
    return kmeans.labels


@save_result
def q8(kmeans: KMeans) -> np.ndarray:
    return kmeans.centroids


@save_result
def q9(kmeans: KMeans) -> float:
    return kmeans.distortion


def get_raw_data(file: str) -> np.ndarray:
    return pd.read_csv(file).to_numpy()


if _CLEANUP:
    def cleanup(file: str, extension: str = '.csv') -> str:
        assert file.endswith(extension)

        df = pd.read_csv(file)

        # type: ignore[arg-type]
        unique_states = np.unique(df["Province_State"].values)

        states = np.array([
            'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California',
            'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia',
            'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa',
            'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
            'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 'Missouri',
            'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey',
            'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio',
            'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina',
            'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
            'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming',
        ])

        not_states = set(unique_states).difference(set(states))

        df = df[~df["Province_State"].isin(not_states)]
        df = df.drop(['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Admin2',
                      'Country_Region', 'Lat', 'Long_', 'Combined_Key'], axis=1)
        df = df.groupby("Province_State").agg(sum)
        df.reset_index(inplace=True)

        file = file.replace(extension, f'_cleaned{extension}')
        df.to_csv(file)
        return file

if __name__ == '__main__':
    if _CLEANUP:
        file = cleanup('time_series_covid19_deaths_US.csv')
    else:
        file = 'time_series_covid19_deaths_US_raw.csv'

    raw_data = get_raw_data(file)
    cum_data = np.cumsum(raw_data, axis=0)

    del raw_data

    data = np.diff(cum_data)

    q1(cum_data)

    del cum_data

    q2(data)

    q3()

    parametric_data = q4(data)

    q5(parametric_data)

    q6(parametric_data)

    kmeans = KMeans(k=5)
    kmeans.fit(parametric_data)

    print(f'KMeans converged in {kmeans.n_iters} iterations')

    q7(kmeans)

    q8(kmeans)

    q9(kmeans)
