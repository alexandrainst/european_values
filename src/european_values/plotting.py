"""Plotting functions."""

import logging
import warnings
from pathlib import Path

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

    # Get the country groupings, which depends on whether we are working with countries
    # or country groups
    country_grouping_str = (
        "country_group" if config.use_country_groups else "country_code"
    )
    unique_country_groupings = (
        survey_df.country_group.unique()
        if config.use_country_groups
        else survey_df.country_code.unique()
    )

    if config.plotting.fast:
        logger.info("Fast UMAP mode enabled, which is non-deterministic but faster.")
    logger.info("Reducing to two dimensions with UMAP...")
    umap = UMAP(
        n_components=2,
        n_neighbors=config.plotting.umap_neighbours,
        random_state=4242 if not config.plotting.fast else None,
        n_jobs=-1 if config.plotting.fast else 1,
    )
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
    )
    if num_importances := config.plotting.top_umap_importances > 0:
        most_important_questions = most_important_questions[:num_importances]

    # Get the average values for the most important questions in the focus group, if
    # available
    if config.focus in survey_df[country_grouping_str].unique():
        focus_mean_values = (
            survey_df.query(f"{country_grouping_str} == @config.focus")
            .loc[:, [q for q, _ in most_important_questions]]
            .mean()
            .tolist()
        )
        focus_stderr_values = (
            survey_df.query(f"{country_grouping_str} == @config.focus")
            .loc[:, [q for q, _ in most_important_questions]]
            .sem()
            .tolist()
        )
        non_focus_mean_values = (
            survey_df.query(f"{country_grouping_str} != @config.focus")
            .loc[:, [q for q, _ in most_important_questions]]
            .mean()
            .tolist()
        )
        non_focus_stderr_values = (
            survey_df.query(f"{country_grouping_str} != @config.focus")
            .loc[:, [q for q, _ in most_important_questions]]
            .sem()
            .tolist()
        )
        logger.info(
            "Most important questions based on UMAP feature importances:\n\t- "
            + "\n\t- ".join(
                [
                    f"{question}: {importance:.4f} "
                    f"({config.focus}: {focus_mean:.2%} ± {1.96 * focus_stderr:.2%}, "
                    f"non-{config.focus}: {non_focus_mean:.2%} ± "
                    f"{1.96 * non_focus_stderr:.2%})"
                    for (
                        (question, importance),
                        focus_mean,
                        focus_stderr,
                        non_focus_mean,
                        non_focus_stderr,
                    ) in zip(
                        most_important_questions,
                        focus_mean_values,
                        focus_stderr_values,
                        non_focus_mean_values,
                        non_focus_stderr_values,
                    )
                ]
            )
        )
    else:
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
        shape=(len(unique_country_groupings), embedding_matrix.shape[1])
    )
    for country_idx, country_grouping in enumerate(unique_country_groupings):
        country_indices = survey_df.query(
            f"{country_grouping_str} == @country_grouping"
        ).index.tolist()
        country_embedding_matrix[country_idx, :] = np.mean(
            embedding_matrix[country_indices, :], axis=0
        )

    logger.info("Creating scatter plot with matplotlib...")
    ax = plt.figure(figsize=(10, 8)).add_subplot(111)
    for country_idx, country_grouping in enumerate(unique_country_groupings):
        colour = plt.cm.tab20(country_idx / len(unique_country_groupings))  # type: ignore[attr-defined]
        country_indices = survey_df.query(
            f"{country_grouping_str} == @country_grouping"
        ).index.tolist()
        if config.plotting.ellipses:
            confidence_ellipse(
                x=embedding_matrix[country_indices, 0],
                y=embedding_matrix[country_indices, 1],
                ax=ax,
                n_std=config.plotting.ellipse_std,
                facecolor="none",
                edgecolor=colour,
            )
        ax.text(
            x=country_embedding_matrix[country_idx, 0],
            y=country_embedding_matrix[country_idx, 1],
            s=country_grouping,
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
    if config.plotting.ellipses:
        ax.set_title(
            f"UMAP projection with ellipse radii = {config.plotting.ellipse_std}σ",
            fontsize=20,
        )
    else:
        ax.set_title("UMAP projection", fontsize=20)

    # Save the plot if configured to do so. We do not overwrite existing files, and
    # instead create a new file with an incremented version number.
    if config.plotting.save_plot:
        output_path = Path("gfx", f"umap_projection_seed{config.seed}.png")
        version = 1
        while output_path.exists():
            version += 1
            output_path = output_path.with_name(
                f"umap_projection_seed{config.seed}_v{version}.png"
            )
        plt.savefig(output_path.as_posix(), dpi=200, bbox_inches="tight")
        logger.info(f"Saved plot to {output_path.as_posix()!r}")

    if config.plotting.show_plot:
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
