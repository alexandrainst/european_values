"""Optimise the questions to be used in the survey.

Usage:
    uv run src/scripts/optimise_survey.py <config_key>=<config_value> ...
"""

import logging

import hydra
import pandas as pd
from datasets import Dataset, load_dataset
from omegaconf import DictConfig

from european_values.data_processing import process_data
from european_values.optimisation import optimise_survey

logger = logging.getLogger("optimise_survey")


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

    logger.info("Processing the data...")
    df = process_data(df=df, config=config)
    logger.info(f"Shape of the data after processing: {df.shape}")

    optimise_survey(survey_df=df, config=config.optimisation)


if __name__ == "__main__":
    main()
