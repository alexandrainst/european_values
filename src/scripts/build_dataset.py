"""Build the dataset.

Usage:
    uv run src/scripts/build_dataset.py <config_key>=<config_value> ...
"""

import logging

import hydra
import numpy as np
import plotly.express as px
from omegaconf import DictConfig
from sklearn.impute import KNNImputer
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

    # Create a 2-dimensional embedding of the EVS trend data. Here we drop the questions
    # which have missing values.
    embedding_matrix = KNNImputer(n_neighbors=25, weights="distance").fit_transform(
        evs_trend_df.iloc[:, 3:]
    )
    embedding_matrix = UMAP(n_components=2).fit_transform(embedding_matrix)
    assert isinstance(embedding_matrix, np.ndarray)
    logger.info(f"Shape of the embedding matrix: {embedding_matrix.shape}")

    # Make a scatter plot of the 2D embedding, where the country codes are colored
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
