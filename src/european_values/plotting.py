"""Plotting functions."""

import logging
import typing as t
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse, Patch
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm
from umap import UMAP

from .constants import (
    AFRICAN_COUNTRY_CODES,
    ASIAN_COUNTRY_CODES,
    EUROPEAN_COUNTRY_CODES,
    MIDDLE_EASTERN_COUNTRY_CODES,
    NORTH_AMERICAN_COUNTRY_CODES,
    OCEANIA_COUNTRY_CODES,
    SOUTH_AMERICAN_COUNTRY_CODES,
)

logger = logging.getLogger(__name__)


warnings.filterwarnings(action="ignore", category=FutureWarning, module="sklearn")


def create_scatter(
    survey_df: pd.DataFrame,
    dimensionality_reduction: t.Literal["umap", "pca"],
    dataset_name: str,
) -> None:
    """Create a scatter plot of the survey data.

    Args:
        survey_df:
            The survey data.
        dimensionality_reduction:
            The dimensionality reduction class to use. Can be either "umap" or "pca".
        dataset_name:
            The name of the dataset to use for the plot title.
    """
    logger.info(f"Shape of the data: {survey_df.shape}")

    def group_country(country_code: str) -> str:
        """Group countries into European Union and Non-European Union."""
        if country_code in EUROPEAN_COUNTRY_CODES:
            return "Europe"
        elif country_code in NORTH_AMERICAN_COUNTRY_CODES:
            return "North America"
        elif country_code in SOUTH_AMERICAN_COUNTRY_CODES:
            return "South America"
        elif country_code in MIDDLE_EASTERN_COUNTRY_CODES:
            return "Middle East"
        elif country_code in AFRICAN_COUNTRY_CODES:
            return "Africa"
        elif country_code in ASIAN_COUNTRY_CODES:
            return "Asia"
        elif country_code in OCEANIA_COUNTRY_CODES:
            return "Oceania"
        else:
            return country_code

    survey_df["country_group"] = survey_df.country_code.apply(group_country)

    # Remove questions for which a country group exists that have not answered the
    # question
    questions_with_missing_answers: dict[str, list[str]] = defaultdict(list)
    for country_group in survey_df.country_group.unique():
        country_df = survey_df.query("country_group == @country_group")
        na_df = country_df.isna().all(axis=0)
        assert isinstance(na_df, pd.Series)
        na_df = na_df[na_df]
        assert isinstance(na_df, pd.Series)
        questions = na_df.index.tolist()
        for question in questions:
            assert isinstance(question, str)
            questions_with_missing_answers[question].append(country_group)

    # Remove the questions where at least one country group has not answered
    survey_df = survey_df.drop(columns=list(questions_with_missing_answers.keys()))
    if questions_with_missing_answers:
        questions_removed_str = "\n\t- ".join(questions_with_missing_answers.keys())
        logger.info(
            f"Removed {len(questions_with_missing_answers)} questions where at least "
            f"one country group has not answered:\n\t- {questions_removed_str}"
        )
        logger.info(
            f"Shape of the data after removing questions with missing answers: "
            f"{survey_df.shape}"
        )

    # Sample for quicker testing
    survey_df = survey_df.sample(n=100_000, random_state=42).reset_index(drop=True)

    # Impute missing values
    question_columns = [col for col in survey_df.columns if col.startswith("question_")]
    embedding_matrix = np.empty(shape=(survey_df.shape[0], len(question_columns)))
    imputer = KNNImputer(n_neighbors=10, weights="distance", keep_empty_features=True)
    for country_group in tqdm(
        iterable=survey_df.country_group.unique(),
        desc="Imputing missing values",
        unit="country group",
    ):
        country_df = survey_df.query("country_group == @country_group")
        country_df = country_df[question_columns].copy()
        assert isinstance(country_df, pd.DataFrame)
        country_embedding = imputer.fit_transform(X=country_df)
        assert isinstance(country_embedding, np.ndarray)
        embedding_matrix[country_df.index, :] = country_embedding
    logger.info(f"Shape of the imputed data: {embedding_matrix.shape}")

    # Normalize the data
    logger.info("Normalizing the data...")
    embedding_matrix = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        X=embedding_matrix
    )

    # Create a 2-dimensional embedding of the data
    logger.info(
        f"Reducing to two dimensions with {dimensionality_reduction.upper()}..."
    )
    reducer_class = UMAP if dimensionality_reduction == "umap" else PCA
    embedding_matrix = reducer_class(n_components=2).fit_transform(embedding_matrix)
    assert isinstance(embedding_matrix, np.ndarray)

    # Create a matrix with mean values for each country group
    country_embedding_matrix = np.empty(
        shape=(survey_df.country_group.nunique(), embedding_matrix.shape[1])
    )
    for country_idx, country_group in enumerate(survey_df.country_group.unique()):
        country_indices = survey_df.query(
            "country_group == @country_group"
        ).index.tolist()
        country_embedding_matrix[country_idx, :] = np.mean(
            embedding_matrix[country_indices, :], axis=0
        )

    logger.info("Creating scatter plot with matplotlib...")
    ax = plt.figure(figsize=(10, 8)).add_subplot(111)
    for country_idx, country_group in enumerate(survey_df.country_group.unique()):
        colour = plt.cm.tab20(country_idx / survey_df.country_group.nunique())
        country_indices = survey_df.query(
            "country_group == @country_group"
        ).index.tolist()
        confidence_ellipse(
            x=embedding_matrix[country_indices, 0],
            y=embedding_matrix[country_indices, 1],
            ax=ax,
            n_std=0.5,
            facecolor="none",
            edgecolor=colour,
        )
        ax.text(
            x=country_embedding_matrix[country_idx, 0],
            y=country_embedding_matrix[country_idx, 1],
            s=country_group,
            fontsize=12,
            ha="center",
            va="center",
            color=colour,
        )
    ax.scatter(
        x=country_embedding_matrix[:, 0], y=country_embedding_matrix[:, 1], alpha=0.0
    )
    ax.set_title(
        f"{dimensionality_reduction.upper()} projection of the {dataset_name}",
        fontsize=20,
    )
    plt.show()


def confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax: Axes,
    n_std: float = 1.0,
    facecolor: str = "none",
    **ellipse_kwargs,
) -> Patch:
    """Create a plot of the covariance confidence ellipse of x- and y-data.

    Args:
        x:
            The x-data to plot.
        y:
            The y-data to plot.
        ax:
            The matplotlib axes to plot on.
        n_std (optional):
            The number of standard deviations to determine the ellipse's radii.
            Defaults to 3.0.
        facecolor (optional):
            The face color of the ellipse, or 'none' for no fill. Defaults to 'none'.
        **ellipse_kwargs:
            Additional keyword arguments to pass to the Ellipse constructor.

    Returns:
        The matplotlib patch object for the ellipse.

    Raises:
        ValueError:
            If x and y are not the same size.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    # Compute the Pearson correlation coefficient
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Using a special case to obtain the eigenvalues of this two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        xy=(0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **ellipse_kwargs,
    )

    # Calculate the standard deviation of x from the squareroot of the variance and
    # multiplying with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x).item()

    # Calculate the standard deviation of y
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y).item()

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)
