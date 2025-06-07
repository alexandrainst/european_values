"""Create plots with the data.

Usage:
    uv run src/scripts/create_plot.py <config_key>=<config_value> ...
"""

import logging

import hydra
import pandas as pd
from datasets import Dataset, load_dataset
from omegaconf import DictConfig

from european_values.data_processing import process_data
from european_values.plotting import create_scatter

logger = logging.getLogger("create_plot")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main function.

    Args:
        config:
            The Hydra config for your project.
    """
    logger.info("Loading the EVS trend data from 1981 to 2017...")
    evs_trend = load_dataset(
        path=config.repo_id, name="evs_trend_data_1981_2017", split="train"
    )
    assert isinstance(evs_trend, Dataset)
    evs_trend_df = evs_trend.to_pandas()
    assert isinstance(evs_trend_df, pd.DataFrame)

    logger.info("Loading the EVS/WVS data from 2017 to 2022...")
    evs_wvs = load_dataset(
        path=config.repo_id, name="evs_wvs_data_2017_2022", split="train"
    )
    assert isinstance(evs_wvs, Dataset)
    evs_wvs_df = evs_wvs.to_pandas()
    assert isinstance(evs_wvs_df, pd.DataFrame)

    logger.info("Combining the EVS trend and EVS/WVS data...")
    df = pd.concat([evs_trend_df, evs_wvs_df], ignore_index=True)

    if config.countries != "all":
        logger.info(f"Filtering data for countries: {config.countries}")
        df = df.query("country_code in @config.countries").reset_index(drop=True)
        logger.info(f"Shape of the data after filtering: {df.shape}")

    logger.info("Processing the data...")
    df = process_data(df=df)
    logger.info(f"Shape of the data after processing: {df.shape}")

    logger.info("Creating the scatter plot...")
    create_scatter(survey_df=df, config=config)


if __name__ == "__main__":
    main()
