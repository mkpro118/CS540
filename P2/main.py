import math
import dataclasses
import numpy as np
from typing import Optional, Any
from functools import wraps


class DecisionTreeClassifier:
    entropy_modes = {
        'shannon': 2,
        'natural': math.e,
        'hartley': 10,
    }

    @dataclasses.dataclass
    class Node:
        feature: Any = None
        threshold: Any = None
        left: Optional['DecisionTreeClassifier.Node'] = None
        right: Optional['DecisionTreeClassifier.Node'] = None
        _: dataclasses.KW_ONLY
        label: Any = None
        values_and_counts: Optional[tuple] = None
        reason: Optional[str] = None

        def _is_leaf(self):
            return (self.label is not None)

        def _max_depth(self):
            if self._is_leaf():
                return 0

            left_depth = self.left._max_depth() if self.left else 0
            right_depth = self.right._max_depth() if self.right else 0

            return max(left_depth, right_depth) + 1

        def __str__(self):
            if self._is_leaf():
                return (
                    f'Node('
                    f'feature={self.feature}, '
                    f'threshold={self.threshold}, '
                    f'left={self.left}, '
                    f'right={self.right}, '
                )
            return (
                f'Node('
                f'label={self.label}, '
                f'values_and_counts={self.values_and_counts}, '
                f'reason={self.reason})'
            )

    def __init__(self, min_split: int = 2,
                 max_depth: int = 64, n_features: Optional[int] = None):
        self.min_split = min_split
        self.max_depth = max_depth
        self.n_features = n_features

    def fit(self, X: np.ndarray, y: np.ndarray,
            feature_idxs: Optional[np.ndarray] = None) -> 'DecisionTreeClassifier':
        assert X.ndim == 2, 'X must be of shape (n_samples, n_features)'

        y = np.ravel(y)
        assert y.shape[0] == X.shape[0], 'y must be of shape (n_samples,)'

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y)

        self.labels = np.unique(y)
        self.n_features = min(
            X.shape[-1], self.n_features) if self.n_features else X.shape[-1]

        self.rng = np.random.default_rng()

        self._info_gain_per_split: list = []
        self._build_tree(X, y)
        return self

    def _build_tree(self, X: np.ndarray, y: np.ndarray):
        self.root = self.__build_tree(X, y)

    def __build_tree(self, X: np.ndarray, y: np.ndarray, *, depth: int = 0):
        n_samples, n_features = X.shape
        n_labels = np.unique(y).shape[0]

        if n_labels == 1:
            return DecisionTreeClassifier.Node(
                label=y[0],
                values_and_counts=np.unique(y, return_counts=True),
                reason='n_labels = 1'
            )

        if n_features == 0:
            values, counts = np.unique(y, return_counts=True)
            return DecisionTreeClassifier.Node(
                label=values[np.argmax(counts)],
                values_and_counts=(values, counts),
                reason='n_features = 0'
            )

        if depth >= self.max_depth:
            values, counts = np.unique(y, return_counts=True)
            return DecisionTreeClassifier.Node(
                label=values[np.argmax(counts)],
                values_and_counts=(values, counts),
                reason='max_depth'
            )

        if n_samples < self.min_split:
            values, counts = np.unique(y, return_counts=True)
            return DecisionTreeClassifier.Node(
                label=values[np.argmax(counts)],
                values_and_counts=(values, counts),
                reason=f'n_samples < self.min_split ({n_samples} < {self.min_split})'
            )

        feature_idxs = np.asarray(self.rng.choice(
            n_features, self.n_features, replace=False)).astype(int)

        best_feature, best_threshold, info_gain = self._best_split(X, y,
                                                                   feature_idxs)

        self._info_gain_per_split.append(info_gain)

        if best_feature is None or best_threshold is None:
            values, counts = np.unique(y, return_counts=True)
            return DecisionTreeClassifier.Node(
                label=values[np.argmax(counts)],
                values_and_counts=(values, counts),
                reason='no optimal split'
            )

        left_idxs, right_idxs = DecisionTreeClassifier._split(
            X[:, best_feature], best_threshold)

        if np.all(left_idxs) or np.all(right_idxs):
            values, counts = np.unique(y, return_counts=True)
            return DecisionTreeClassifier.Node(
                label=values[np.argmax(counts)],
                values_and_counts=(values, counts),
                reason='further splits are meaningless'
            )

        left = self.__build_tree(
            X[left_idxs, :], y[left_idxs], depth=depth + 1)
        right = self.__build_tree(
            X[right_idxs, :], y[right_idxs], depth=depth + 1)

        return DecisionTreeClassifier.Node(best_feature, best_threshold, left, right)

    def _best_split(self, X: np.ndarray, y: np.ndarray,
                    feature_idxs: np.ndarray) -> tuple[Optional[int], Optional[Any], float]:
        best_info_gain = -1.
        split_idx, split_threshold = None, None

        for feature_idx in feature_idxs:
            X_col = X[:, feature_idx]
            thresholds = np.unique(X_col)

            for threshold in thresholds:
                info_gain = DecisionTreeClassifier.info_gain(
                    X_col, y, threshold)

                if info_gain <= best_info_gain:
                    continue

                best_info_gain = info_gain
                split_idx = feature_idx
                split_threshold = threshold

        return split_idx, split_threshold, best_info_gain

    def predict(self, X: np.ndarray) -> Any:
        return np.asarray(
            [DecisionTreeClassifier._predict_traverse(self.root, x) for x in X]
        )

    def summary(self, show_counts=False, show_reason=False, indent=2):
        return DecisionTreeClassifier._summary_traverse(
            self.root,
            show_counts=show_counts,
            show_reason=show_reason,
            indent=indent
        )

    def get_max_depth(self):
        return self.root._max_depth()

    @staticmethod
    def _summary_traverse(node: Node, depth: int = 0,
                          show_counts=False, show_reason=False, indent=2) -> str:
        if node._is_leaf():
            # type: ignore[misc]
            counts = ', '.join(map(str, zip(*node.values_and_counts)))
            return (
                f'{(" " * indent) * depth}return {node.label}' + (
                    f' | reason: {node.reason}' if show_reason else '') + (
                    f' | counts: {counts}\n' if show_counts else '\n')
            )

        summary = f'{(" " * indent) * depth}if x{node.feature} <= {node.threshold}\n'

        assert node.left, str(node)  # For mypy
        summary += DecisionTreeClassifier._summary_traverse(
            node.left, depth + 1, show_counts=show_counts,
            show_reason=show_reason, indent=indent
        )

        summary += f'{(" " * indent) * depth}else\n'

        assert node.right, str(node)  # For mypy
        summary += DecisionTreeClassifier._summary_traverse(
            node.right, depth + 1, show_counts=show_counts,
            show_reason=show_reason, indent=indent
        )

        return summary

    @staticmethod
    def _predict_traverse(node: Node, x: np.ndarray) -> Any:
        if node._is_leaf():
            return node.label

        if x[node.feature] <= node.threshold:
            assert node.left, 'Invalid DecisionTreeClassifier!'
            return DecisionTreeClassifier._predict_traverse(node.left, x)

        assert node.right, 'Invalid DecisionTreeClassifier!'
        return DecisionTreeClassifier._predict_traverse(node.right, x)

    @staticmethod
    def _split(X: np.ndarray, threshold: Any) -> tuple[np.ndarray, np.ndarray]:
        return X <= threshold, X > threshold

    @staticmethod
    def info_gain(feature_col: np.ndarray,
                  labels: np.ndarray, threshold: Any) -> float:
        current_entropy = DecisionTreeClassifier.entropy(labels)

        left_idxs, right_idxs = DecisionTreeClassifier._split(
            feature_col, threshold)

        if any((not len(left_idxs), not len(right_idxs))):
            return 0.

        n = len(labels)

        left = labels[left_idxs]
        right = labels[right_idxs]

        left_entropy = DecisionTreeClassifier.entropy(left)
        right_entropy = DecisionTreeClassifier.entropy(right)

        left_weight = len(left) / n
        right_weight = len(right) / n

        split_entropy = np.sum((left_weight * left_entropy,
                                right_weight * right_entropy))  # type: ignore

        info_gain = current_entropy - split_entropy

        return info_gain

    @staticmethod
    def entropy(labels: np.ndarray, *,
                base: Optional[float] = None,
                mode: Optional[str] = None) -> float:
        assert labels.ndim == 1, 'labels must be of shape (n_samples,)'

        if base:
            assert not mode, 'only one of base or mode should be provided'
            assert base > 0, 'base should be a positive number'
        elif mode:
            base = DecisionTreeClassifier.entropy_modes.get(mode)
            assert base, (
                f'invalid {mode=}, valid modes are '
                f'{", ".join(DecisionTreeClassifier.entropy_modes.keys())}'
            )
        else:
            base = 2

        probs = np.unique(labels, return_counts=True)[1] / labels.shape[0]
        return -sum((prob * math.log(prob, base) for prob in probs))


def save_result(file):
    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            result = func(*args, **kwargs)
            with open(file, 'w') as f:
                f.write(result)
            return result
        return inner
    return decorator


def map_result(func):
    @wraps(func)
    def inner(*args, **kwargs):
        result = func(*args, **kwargs)
        _map = {
            'x0': 'x3',
            'x1': 'x5',
            'x2': 'x4',
            'x3': 'x2',
            'x4': 'x9',
            'x5': 'x10',
        }

        mapped = ''

        for line in result.splitlines():
            for k in _map.keys():
                if k in line:
                    line = line.replace(k, _map[k])
                    break
            mapped += f'{line}\n'

        return mapped
    return inner


@save_result('q1.txt')
def q1(X: np.ndarray, y: np.ndarray) -> str:
    values, counts = np.unique(y, return_counts=True)
    idx = np.argsort(values)
    counts = counts[idx]
    return f'{counts[0]}, {counts[1]}'


@save_result('q2.txt')
def q2(X: np.ndarray, y: np.ndarray) -> str:
    return f'{DecisionTreeClassifier.entropy(y)}'


@save_result('q3.txt')
def q3(model: DecisionTreeClassifier) -> str:
    import re

    summary = model.summary(show_counts=True)

    pattern = re.compile(r'.+\d+, (\d+).+\d+, (\d+)')

    match = pattern.findall(summary)

    below_benign, below_malignant = match[0]
    above_benign, above_malignant = match[1]

    return ', '.join((below_benign, above_benign,
                      below_malignant, above_malignant))


@save_result('q4.txt')
def q4(data_str: str, labels: np.ndarray) -> str:
    entropy = DecisionTreeClassifier.entropy(labels)

    data = tuple(map(int, data_str.split(', ')))

    n2_, n2 = data[1], data[0]
    n4_, n4 = data[3], data[2]
    n = sum(data)
    nb = n2_ + n4_
    na = n2 + n4
    x = (n2_, n2, n4_, n4)
    y = (nb, na, nb, na)
    cond_entropy = sum((-i * math.log(i / j, 2) for i, j in zip(x, y))) / n

    return f'{entropy - cond_entropy}'


@save_result('q5.txt')
@map_result
def q5(model: DecisionTreeClassifier) -> str:
    return model.summary(indent=2)


@save_result('q6.txt')
def q6(model: DecisionTreeClassifier) -> str:
    return str(model.get_max_depth())


@save_result('q7.txt')
def q7(model: DecisionTreeClassifier, X: np.ndarray) -> str:
    return ','.join(map(str, model.predict(X).astype(int).tolist()))


@save_result('q8.txt')
@map_result
def q8(model: DecisionTreeClassifier) -> str:
    return model.summary(indent=4)


@save_result('q9.txt')
def q9(model: DecisionTreeClassifier, X: np.ndarray) -> str:
    return ','.join(map(str, model.predict(X).astype(int).tolist()))


if __name__ == '__main__':
    # np.random.seed(118)
    data = np.genfromtxt('breast_cancer_wisconsin.csv',
                         delimiter=',', dtype=int)

    required_features = np.asarray([2, 4, 3, 1, 8, 9, -1])

    data = data[:, required_features]
    X, y = data[:, :-1], data[:, -1]

    q1(X, y)
    q2(X, y)
    model = DecisionTreeClassifier(max_depth=1)
    model.fit(X, y)
    data_str = q3(model)
    q4(data_str, y)

    model = DecisionTreeClassifier(min_split=1)
    model.fit(X, y)
    q5(model)
    q6(model)

    test_data = np.genfromtxt('test.csv',
                              delimiter=',', dtype=int)
    test_data = test_data[:, required_features[:-1]]

    q7(model, test_data)

    model = DecisionTreeClassifier(max_depth=6)
    model.fit(X, y)

    q8(model)
    q9(model, test_data)
    print('finished')
