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

    # Only use a subset of questions if specified
    if config.subset_csv is not None:
        subset_df = pd.read_csv(config.subset_csv)
        if "question" in subset_df.columns:
            question_subset = subset_df.question.unique().tolist()
            if config.top_num_questions_in_subset is not None:
                question_subset = question_subset[: config.top_num_questions_in_subset]
        else:
            question_subset = list(
                {line.split(":")[0] for line in subset_df.index.tolist()}
            )
        question_columns_to_remove = [
            col
            for col in df.columns
            if col.startswith("question_") and col not in question_subset
        ]
        df.drop(columns=question_columns_to_remove, inplace=True)
        logger.info(
            f"Removed {len(question_columns_to_remove):,} questions not in the "
            f"specified subset CSV file {config.subset_csv}."
        )
        logger.info(f"Shape of the data after filtering: {df.shape}")

    train_model(
        survey_df=df,
        model_type=config.training.model_type,
        n_cross_val=config.training.n_cross_val,
        n_jobs=config.training.n_jobs,
        n_estimators=config.training.n_estimators,
        seed=config.seed,
        bootstrap=config.training.bootstrap,
        compute_importances=config.training.compute_importances,
    )


if __name__ == "__main__":
    main()
