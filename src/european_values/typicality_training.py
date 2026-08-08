"""Training utilities for typicality-based pipelines."""

from __future__ import annotations

import logging
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from . import typicality_transformers
from .typicality_transformers import ECDFTransformer, KNNDistanceTransformer

logger = logging.getLogger(__name__)


def train_kde_ecdf_pipeline(
    eu_df: pd.DataFrame,
    scaler: MinMaxScaler,
    *,
    bandwidth: float = 0.5,
    output_path: str = "models/typicality_kde_ecdf_pipeline.pkl",
) -> Pipeline:
    """Train a KDE + ECDF pipeline on EU survey data."""
    question_columns = [col for col in eu_df.columns if col.startswith("question_")]
    logger.info(f"Training KDE+ECDF with {len(question_columns):,} questions")

    full_matrix = scaler.transform(eu_df[question_columns].values)

    model = KernelDensity(bandwidth=bandwidth).fit(full_matrix)
    model.transform = model.score_samples.__get__(model)

    log_likelihoods = model.transform(full_matrix)
    scorer = ECDFTransformer(higher_is_better=True).fit(log_likelihoods)

    _log_distribution_stats("KDE log-likelihoods", log_likelihoods)
    _log_distribution_stats("KDE ECDF scores", scorer.transform(log_likelihoods))

    pipeline = Pipeline([("scaler", scaler), ("model", model), ("scorer", scorer)])
    _save_pipeline(pipeline, output_path)
    return pipeline


def train_knn_ecdf_pipeline(
    eu_df: pd.DataFrame,
    scaler: MinMaxScaler,
    *,
    k: int = 50,
    metric: str = "minkowski",
    n_jobs: int | None = None,
    output_path: str = "models/typicality_knn_ecdf_pipeline.pkl",
) -> Pipeline:
    """Train a kNN-distance + ECDF pipeline on EU survey data."""
    question_columns = [col for col in eu_df.columns if col.startswith("question_")]
    logger.info(f"Training kNN+ECDF with {len(question_columns):,} questions")

    full_matrix = scaler.transform(eu_df[question_columns].values)

    knn = KNNDistanceTransformer(k=k, metric=metric, n_jobs=n_jobs)
    distances = knn.fit_transform(full_matrix)
    scorer = ECDFTransformer(higher_is_better=False).fit(distances)

    _log_distribution_stats("kNN distances", distances)
    _log_distribution_stats("kNN ECDF scores", scorer.transform(distances))

    pipeline = Pipeline([("scaler", scaler), ("model", knn), ("scorer", scorer)])
    _save_pipeline(pipeline, output_path)
    return pipeline


def _save_pipeline(pipeline: Pipeline, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cloudpickle.register_pickle_by_value(module=typicality_transformers)
    with path.open("wb") as f:
        cloudpickle.dump(obj=pipeline, file=f)
    logger.info(f"Pipeline saved to {path.as_posix()}")


def _log_distribution_stats(label: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float).reshape(-1)
    logger.info(
        f"{label}:\n"
        f"\t- Mean: {values.mean():.4f}\n"
        f"\t- Std: {values.std():.4f}\n"
        f"\t- Min: {values.min():.4f}\n"
        f"\t- 10% quantile: {np.quantile(values, q=0.1):.4f}\n"
        f"\t- 90% quantile: {np.quantile(values, q=0.9):.4f}\n"
        f"\t- Max: {values.max():.4f}\n"
    )
