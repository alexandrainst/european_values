"""Build the dataset.

Usage:
    uv run src/scripts/build_dataset.py <config_key>=<config_value> ...
"""

import logging

import hydra
import numpy as np
import plotly.express as px
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from umap import UMAP

from european_values.data_loading import load_evs_trend_data

logger = logging.getLogger("build_dataset")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main function.

    Args:
        config:
            The Hydra config for your project.
    """
    evs_trend_df = load_evs_trend_data()

    # TEMP: Focus on a single wave
    evs_trend_df = evs_trend_df.query(
        "year > 1985 and year < 1995 and country_code in ['DK', 'SE', 'NO', 'IS', 'US']"
    )
    logger.info(f"Shape of the sliced data: {evs_trend_df.shape}")

    # Create a 2-dimensional embedding of the EVS trend data
    logger.info("Imputing missing values...")
    embedding_matrix = IterativeImputer(
        estimator=RandomForestClassifier(n_estimators=100, n_jobs=-1),
        skip_complete=True,
        n_nearest_features=10,
        initial_strategy="most_frequent",
        max_iter=20,
        random_state=4242,
    ).fit_transform(evs_trend_df.iloc[:, 3:])
    logger.info(f"Shape of the imputed data: {embedding_matrix.shape}")

    logger.info("Reducing to two dimensions with UMAP...")
    embedding_matrix = UMAP(n_components=2).fit_transform(embedding_matrix)
    assert isinstance(embedding_matrix, np.ndarray)

    # Make a scatter plot of the 2D embedding, where the country codes are colored
    logger.info("Creating scatter plot...")
    fig = px.scatter(
        x=embedding_matrix[:, 0],
        y=embedding_matrix[:, 1],
        color=evs_trend_df.country_code.tolist(),
        title="UMAP projection of the EVS trend data",
        labels=dict(x="UMAP 1", y="UMAP 2", color="Country Code"),
        color_discrete_sequence=px.colors.qualitative.Plotly,
        width=800,
        height=600,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        title_font=dict(size=20), legend_title_font=dict(size=16), font=dict(size=14)
    )
    fig.show()


if __name__ == "__main__":
    main()
