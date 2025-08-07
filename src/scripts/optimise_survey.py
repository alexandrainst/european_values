"""Optimise the questions to be used in the survey.

Usage:
    uv run src/scripts/optimise_survey.py <config_key>=<config_value> ...
"""

import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

from european_values.data_loading import load_evs_trend_data, load_evs_wvs_data
from european_values.data_processing import process_data
from european_values.optimisation import optimise_survey
from european_values.plotting import create_scatter

logger = logging.getLogger("optimise_survey")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main function.

    Args:
        config:
            The Hydra config for your project.
    """
    match (config.include_evs_trend, config.include_evs_wvs):
        case (True, True):
            logger.info("Loading EVS trend and EVS/WVS data...")
            evs_trend_df = load_evs_trend_data()
            evs_wvs_df = load_evs_wvs_data()
            df = pd.concat([evs_trend_df, evs_wvs_df], ignore_index=True)
        case (True, False):
            logger.info("Loading only EVS trend data...")
            df = load_evs_trend_data()
        case (False, True):
            logger.info("Loading only EVS/WVS data...")
            df = load_evs_wvs_data()
        case _:
            raise ValueError(
                "At least one of `include_evs_trend` or `include_evs_wvs` must be True."
            )

    logger.info("Processing the data...")
    df, scaler = process_data(df, config)
    logger.info(f"Shape of the data after processing: {df.shape}")

    df = optimise_survey(survey_df=df, config=config)

    logger.info("Creating the scatter plot...")
    config.plotting.fast = True  # Set fast mode for plotting
    create_scatter(survey_df=df, config=config)


if __name__ == "__main__":
    main()
