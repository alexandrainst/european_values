"""Transformers for typicality scoring."""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class ECDFTransformer:
    """Map scores to an empirical CDF percentile.

    If higher_is_better is False, the percentile is inverted so that higher values
    correspond to higher typicality.
    """

    def __init__(self, *, higher_is_better: bool = True) -> None:
        self.higher_is_better = higher_is_better

    def fit(self, X: np.ndarray, y: None = None) -> "ECDFTransformer":
        values = np.asarray(X, dtype=float).reshape(-1)
        if values.size == 0:
            raise ValueError("ECDFTransformer cannot be fit on empty data.")
        self.sorted_values_ = np.sort(values)
        self.n_ = values.size
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "sorted_values_"):
            raise ValueError("ECDFTransformer must be fitted before calling transform.")
        values = np.asarray(X, dtype=float).reshape(-1)
        ranks = np.searchsorted(self.sorted_values_, values, side="right")
        percentiles = ranks / self.n_
        if not self.higher_is_better:
            percentiles = 1.0 - percentiles
        return percentiles


class KNNDistanceTransformer:
    """Compute kNN distance for typicality scoring.

    During fit_transform (training data), it drops the closest neighbor to avoid
    the trivial zero-distance self match.
    """

    def __init__(
        self,
        *,
        k: int = 50,
        metric: str = "minkowski",
        n_jobs: int | None = None,
    ) -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: None = None) -> "KNNDistanceTransformer":
        X = np.asarray(X, dtype=float)
        self._nn = NearestNeighbors(
            n_neighbors=self.k + 1, metric=self.metric, n_jobs=self.n_jobs
        )
        self._nn.fit(X)
        return self

    def fit_transform(self, X: np.ndarray, y: None = None) -> np.ndarray:
        self.fit(X, y=y)
        return self._distances(np.asarray(X, dtype=float), exclude_self=True)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_nn"):
            raise ValueError(
                "KNNDistanceTransformer must be fitted before calling transform."
            )
        return self._distances(np.asarray(X, dtype=float), exclude_self=False)

    def _distances(self, X: np.ndarray, *, exclude_self: bool) -> np.ndarray:
        distances, _ = self._nn.kneighbors(X, n_neighbors=self.k + 1)
        if exclude_self:
            # Drop the closest neighbor (self) and take the k-th distance.
            distances = distances[:, 1:]
        return distances[:, self.k - 1]
