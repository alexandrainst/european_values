"""Train a generative model on the data."""

import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

from european_values.data_loading import load_evs_trend_data, load_evs_wvs_data
from european_values.data_processing import process_data
from european_values.generative_training import train_generative_model

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

    # Apply subset filtering
    if config.subset_csv is not None:
        subset_df = pd.read_csv(config.subset_csv)
        question_subset = (
            subset_df.question.unique().tolist()
            if "question" in subset_df.columns
            else list({line.split(":")[0] for line in subset_df.index.tolist()})
        )
        question_cols_to_remove = [
            col
            for col in df.columns
            if col.startswith("question_") and col not in question_subset
        ]
        df.drop(columns=question_cols_to_remove, inplace=True)
        logger.info(
            f"Using {len(question_subset)} questions from the subset "
            f"{config.subset_csv!r}."
        )

    # Process data but SKIP normalization (let pipeline handle it)
    logger.info("Processing the data WITHOUT normalization...")
    df, scaler = process_data(df=df, config=config, normalize=False)

    # Train the model
    train_generative_model(
        eu_df=df.query("country_group == 'EU'"),
        scaler=scaler,
        test_samples_per_country=config.generative_training.test_samples_per_country,
        seed=config.seed,
    )


if __name__ == "__main__":
    main()
