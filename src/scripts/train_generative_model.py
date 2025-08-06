"""Train a generative model on the data."""

import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

from european_values.data_loading import load_evs_wvs_data
from european_values.data_processing import process_data
from european_values.generative_training import train_generative_model

logger = logging.getLogger("train_generative_model")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main function."""
    # Load data
    logger.info("Loading only EVS/WVS data...")
    df = load_evs_wvs_data()

    # Process data but SKIP normalization (let pipeline handle it)
    logger.info("Processing the data WITHOUT normalization...")
    df, _ = process_data(df=df, config=config, normalize=False)

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
        logger.info(f"Using {len(question_subset)} questions from subset")

    # Train the model
    train_generative_model(
        survey_df=df,
        max_components=config.generative_training.max_components,
        samples_per_country_val_test=config.generative_training.samples_per_country_val_test,
        seed=config.seed,
    )


if __name__ == "__main__":
    main()
