"""Plotting functions."""

import logging
import typing as t

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from umap import UMAP

logger = logging.getLogger(__name__)


def create_scatter(
    survey_df: pd.DataFrame,
    slice_query: str | None,
    max_imputation_iterations: int,
    dimensionality_reduction: t.Literal["umap", "pca"] = "umap",
) -> None:
    """Create a scatter plot of the survey data.

    Args:
        survey_df:
            The survey data.
        slice_query:
            The query to slice the data, compatible with DataFrame.query(). If None,
            the data will not be sliced.
        max_imputation_iterations:
            The maximum number of iterations for the imputer.
        dimensionality_reduction:
            The dimensionality reduction class to use. Can be either "umap" or "pca".
    """
    logger.info(f"Shape of the data: {survey_df.shape}")

    if slice_query:
        logger.info(f"Slicing data with query: {slice_query}")
        survey_df = survey_df.query(slice_query)
        logger.info(f"Shape of the sliced data: {survey_df.shape}")

    # Create a 2-dimensional embedding of the EVS trend data
    logger.info("Imputing missing values...")
    embedding_matrix = IterativeImputer(
        estimator=RandomForestClassifier(n_estimators=100, n_jobs=-1),
        skip_complete=True,
        n_nearest_features=10,
        initial_strategy="most_frequent",
        max_iter=max_imputation_iterations,
        random_state=4242,
    ).fit_transform(survey_df.iloc[:, 3:])
    logger.info(f"Shape of the imputed data: {embedding_matrix.shape}")

    logger.info(
        f"Reducing to two dimensions with {dimensionality_reduction.upper()}..."
    )
    reducer_class = UMAP if dimensionality_reduction == "umap" else PCA
    embedding_matrix = reducer_class(n_components=2).fit_transform(embedding_matrix)
    assert isinstance(embedding_matrix, np.ndarray)

    # Make a scatter plot of the 2D embedding, where the country codes are colored
    logger.info("Creating scatter plot...")
    fig = px.scatter(
        x=embedding_matrix[:, 0],
        y=embedding_matrix[:, 1],
        color=survey_df.country_code.tolist(),
        title=f"{dimensionality_reduction.upper()} projection of the EVS trend data",
        labels=dict(
            x=f"{dimensionality_reduction.upper()} 1",
            y=f"{dimensionality_reduction.upper()} 2",
            color="Country Code",
        ),
        color_discrete_sequence=px.colors.qualitative.Plotly,
        width=800,
        height=600,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        title_font=dict(size=20), legend_title_font=dict(size=16), font=dict(size=14)
    )
    fig.show()
