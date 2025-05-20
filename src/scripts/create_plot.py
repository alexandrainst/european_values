"""Create plots with the data.

Usage:
    uv run src/scripts/create_plot.py <config_key>=<config_value> ...
"""

import logging

import hydra
from omegaconf import DictConfig

from european_values.data_loading import load_evs_trend_data, load_evs_wvs_data
from european_values.plotting import create_scatter

logger = logging.getLogger("create_plot")


@hydra.main(config_path="../../config", config_name="plotting", version_base=None)
def main(config: DictConfig) -> None:
    """Main function.

    Args:
        config:
            The Hydra config for your project.
    """
    if config.data is None:
        raise ValueError(
            "No data selected. Please set `data` to be either 'evs_trend' or 'evs_wvs'."
        )
    elif config.data not in ["evs_trend", "evs_wvs"]:
        raise ValueError(
            f"Invalid data selected: {config.data}. Please set `data` to be either "
            "'evs_trend' or 'evs_wvs'."
        )

    df = load_evs_trend_data() if config.data == "evs_trend" else load_evs_wvs_data()
    create_scatter(survey_df=df, slice_query=config.query)


if __name__ == "__main__":
    main()
