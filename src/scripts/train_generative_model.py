"""Train a generative model on the data."""

import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

from european_values.data_loading import load_evs_trend_data, load_evs_wvs_data
from european_values.data_processing import process_data
from european_values.generative_training import train_generative_model
from european_values.utils import apply_subset_filtering

logger = logging.getLogger("train_generative_model")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main function."""
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

    df = apply_subset_filtering(df=df, subset_csv_path=config.subset_csv)

    logger.info("Processing the data WITHOUT normalization...")
    df, scaler = process_data(df=df, config=config, normalize=False)

    train_generative_model(
        eu_df=df.query("country_group == 'EU'"),
        scaler=scaler,
        test_samples_per_country=config.generative_training.test_samples_per_country,
        seed=config.seed,
        n_components_max=config.generative_training.n_components_max,
        selection_criterion=config.generative_training.selection_criterion,
        covariance_type=config.generative_training.covariance_type,
        n_init=config.generative_training.n_init,
        reg_covar=config.generative_training.reg_covar,
    )


if __name__ == "__main__":
    main()
