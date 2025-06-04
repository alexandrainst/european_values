"""Plotting functions."""

import logging
import typing as t
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse, Patch
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from umap import UMAP

logger = logging.getLogger(__name__)


warnings.filterwarnings(action="ignore", category=FutureWarning, module="sklearn")


def create_scatter(
    survey_df: pd.DataFrame,
    slice_query: str | None,
    dimensionality_reduction: t.Literal["umap", "pca"] = "umap",
) -> None:
    """Create a scatter plot of the survey data.

    Args:
        survey_df:
            The survey data.
        slice_query:
            The query to slice the data, compatible with DataFrame.query(). If None,
            the data will not be sliced.
        dimensionality_reduction:
            The dimensionality reduction class to use. Can be either "umap" or "pca".
    """
    logger.info(f"Shape of the data: {survey_df.shape}")

    if slice_query:
        logger.info(f"Slicing data with query: {slice_query}")
        survey_df = survey_df.query(slice_query)
        logger.info(f"Shape of the sliced data: {survey_df.shape}")

    # Remove questions for which a country exists that have not answered the question
    questions_with_missing_answers: dict[str, list[str]] = defaultdict(list)
    for country_code in survey_df.country_code.unique():
        country_df = survey_df.query("country_code == @country_code")
        na_df = country_df.isna().all(axis=0)
        assert isinstance(na_df, pd.Series)
        na_df = na_df[na_df]
        assert isinstance(na_df, pd.Series)
        questions = na_df.index.tolist()
        for question in questions:
            assert isinstance(question, str)
            questions_with_missing_answers[question].append(country_code)

    # Remove the questions where at least one country has not answered
    survey_df = survey_df.drop(columns=list(questions_with_missing_answers.keys()))
    if questions_with_missing_answers:
        questions_removed_str = "\n\t- ".join(questions_with_missing_answers.keys())
        logger.info(
            f"Removed {len(questions_with_missing_answers)} questions where at least "
            f"one country has not answered:\n\t- {questions_removed_str}"
        )
        logger.info(
            f"Shape of the data after removing questions with missing answers: "
            f"{survey_df.shape}"
        )

    # Remove the questions where everyone has answered the same answer
    questions_with_same_answers: dict[str, list[str]] = defaultdict(list)
    for question in survey_df.columns[3:]:
        unique_answers = survey_df[question].unique()
        if len(unique_answers) == 1:
            questions_with_same_answers[question] = unique_answers.tolist()
    survey_df = survey_df.drop(columns=list(questions_with_same_answers.keys()))
    if questions_with_same_answers:
        questions_removed_str = "\n\t- ".join(
            f"{question} ({', '.join(answers)})"
            for question, answers in questions_with_same_answers.items()
        )
        logger.info(
            f"Removed {len(questions_with_same_answers)} questions where all countries "
            f"have answered the same answer:\n\t- {questions_removed_str}"
        )

    # Impute missing values
    logger.info("Imputing missing values...")
    embedding_matrix = SimpleImputer(strategy="median").fit_transform(
        survey_df.iloc[:, 3:]
    )
    logger.info(f"Shape of the imputed data: {embedding_matrix.shape}")

    # Normalize the data
    logger.info("Normalizing the data...")
    embedding_matrix = MinMaxScaler(feature_range=(0, 1)).fit_transform(
        X=embedding_matrix
    )

    # Create a 2-dimensional embedding of the EVS trend data
    logger.info(
        f"Reducing to two dimensions with {dimensionality_reduction.upper()}..."
    )
    reducer_class = UMAP if dimensionality_reduction == "umap" else PCA
    embedding_matrix = reducer_class(n_components=2).fit_transform(embedding_matrix)
    assert isinstance(embedding_matrix, np.ndarray)

    logger.info("Creating scatter plot with matplotlib...")
    ax = plt.figure(figsize=(10, 8)).add_subplot(111)
    for country_idx, country_code in enumerate(survey_df.country_code.unique()):
        colour = plt.cm.tab20(country_idx / len(survey_df.country_code.unique()))
        country_indices = survey_df.query(
            "country_code == @country_code"
        ).index.tolist()
        confidence_ellipse(
            x=embedding_matrix[country_indices, 0],
            y=embedding_matrix[country_indices, 1],
            ax=ax,
            n_std=1.0,
            facecolor="none",
            edgecolor=colour,
        )
    ax.scatter(
        x=embedding_matrix[:, 0],
        y=embedding_matrix[:, 1],
        c=survey_df.country_code.astype("category").cat.codes,
        cmap="tab20",
        s=5,
    )
    ax.set_title(
        f"{dimensionality_reduction.upper()} projection of the EVS trend data",
        fontsize=20,
    )
    plt.show()

    # Make a scatter plot of the 2D embedding, where the country codes are colored
    logger.info("Creating scatter plot with plotly...")
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


def confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax: Axes,
    n_std: float = 3.0,
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
