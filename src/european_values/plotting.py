"""Plotting functions."""

import logging
import warnings

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Ellipse, Patch
from omegaconf import DictConfig
from pandas.errors import PerformanceWarning
from umap import UMAP

logger = logging.getLogger(__name__)


warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=PerformanceWarning)


def create_scatter(survey_df: pd.DataFrame, config: DictConfig) -> None:
    """Create a scatter plot of the survey data.

    Args:
        survey_df:
            The survey data.
        config:
            The Hydra config.
    """
    # Create the embedding matrix
    logger.info("Creating embedding matrix...")
    question_columns = [col for col in survey_df.columns if col.startswith("question_")]
    embedding_matrix = survey_df[question_columns].values

    logger.info("Reducing to two dimensions with UMAP...")
    umap = UMAP(n_components=2, n_neighbors=config.umap_neighbours, random_state=4242)
    embedding_matrix = umap.fit_transform(embedding_matrix)
    assert isinstance(embedding_matrix, np.ndarray)

    # Get feature importances from UMAP
    logger.info("Calculating feature importances based on UMAP...")
    df_with_umap = pd.concat(
        [survey_df, pd.DataFrame(embedding_matrix, columns=["umap_1", "umap_2"])],
        axis=1,
    )
    importances: dict[str, float] = dict()
    for question in question_columns:
        importance = (
            df_with_umap[["umap_1", "umap_2", question]].corr().iloc[:2, 2].abs().mean()
        )
        importances[question] = importance
    most_important_questions = sorted(
        importances.items(), key=lambda item: item[1], reverse=True
    )[: config.top_umap_importances]
    logger.info(
        "Most important questions based on UMAP feature importances:\n\t- "
        + "\n\t- ".join(
            [
                f"{question}: {importance:.4f}"
                for question, importance in most_important_questions
            ]
        )
    )

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
        colour = plt.cm.tab20(country_idx / survey_df.country_group.nunique())  # type: ignore[attr-defined]
        country_indices = survey_df.query(
            "country_group == @country_group"
        ).index.tolist()
        if config.ellipses:
            confidence_ellipse(
                x=embedding_matrix[country_indices, 0],
                y=embedding_matrix[country_indices, 1],
                ax=ax,
                n_std=config.ellipse_std,
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
    # We create an invisible scatter plot to set the limits of the axes, which is
    # required to display the ellipses correctly
    ax.scatter(
        x=country_embedding_matrix[:, 0], y=country_embedding_matrix[:, 1], alpha=0.0
    )
    ax.set_title("UMAP projection", fontsize=20)
    plt.show()


def confidence_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    ax: Axes,
    n_std: float,
    facecolor: str,
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
        facecolor (optional):
            The face color of the ellipse, or 'none' for no fill.
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
