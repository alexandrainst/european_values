"""Script to evaluate LLM benchmark."""

import logging

import hydra
import joblib
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from european_values.data_loading import load_evs_trend_data, load_evs_wvs_data
from european_values.data_processing import process_data

logger = logging.getLogger("evaluate_llm")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main evaluation function."""
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
        logger.info(f"Using {len(question_subset)} questions from subset")

    # Process data without normalization (let pipeline handle it)
    logger.info("Processing the data WITHOUT normalization...")
    df, _ = process_data(df=df, config=config, normalize=False)

    # Run evaluation
    logger.info("Running evaluation...")
    pipeline = joblib.load(config.evaluation.model_path)
    question_cols = [col for col in df.columns if col.startswith("question_")]
    for country_group in df.country_group.unique():
        group_df = df.query("country_group == @country_group")
        responses = group_df[question_cols].values
        log_likelihoods = pipeline.score_samples(responses)

        # We normalise so that anything below -100 is 0% and anything above 0 is 100%,
        # with a linear scale in between.
        normalised_scores = (log_likelihoods + 100) / 100
        normalised_scores = np.clip(normalised_scores, 0, 1)

        logger.info(
            f"Log-likelihoods for {country_group}:\n"
            f"\t- Mean: {log_likelihoods.mean():.2f}\n"
            f"\t- Std: {log_likelihoods.std():.2f}\n"
            f"\t- Min: {log_likelihoods.min():.2f}\n"
            f"\t- 10% quantile: {np.quantile(log_likelihoods, q=0.1):.2f}\n"
            f"\t- 90% quantile: {np.quantile(log_likelihoods, q=0.9):.2f}\n"
            f"\t- Max: {log_likelihoods.max():.2f}\n"
            f"\t- Mean normalised score: {normalised_scores.mean():.2%} "
        )


if __name__ == "__main__":
    main()
