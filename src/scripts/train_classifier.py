"""Train a classifier on the data.

Usage:
    uv run src/scripts/train_classifier.py <config_key>=<config_value> ...
"""

import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

from european_values.data_loading import load_evs_trend_data, load_evs_wvs_data
from european_values.data_processing import process_data
from european_values.training import train_model

logger = logging.getLogger("train_classifier")


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
    df = process_data(df=df, config=config)
    logger.info(f"Shape of the data after processing: {df.shape}")

    train_model(
        survey_df=df,
        model_type=config.training.model_type,
        n_cross_val=config.training.n_cross_val,
        n_jobs=config.training.n_jobs,
        n_estimators=config.training.n_estimators,
        seed=config.seed,
    )


if __name__ == "__main__":
    main()
