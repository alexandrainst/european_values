"""Script to evaluate LLM benchmark."""

import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

from european_values.data_loading import load_evs_wvs_data
from european_values.data_processing import process_data
from european_values.llm_evaluation import evaluate_survey_data

logger = logging.getLogger("evaluate_llm")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main evaluation function."""
    # Load and process data
    logger.info("Loading data...")
    df = load_evs_wvs_data()
    df = process_data(df=df, config=config)

    # Apply subset if specified
    if config.subset_csv is not None:
        subset_df = pd.read_csv(config.subset_csv)
        if "question" in subset_df.columns:
            question_subset = subset_df.question.unique().tolist()
        else:
            question_subset = list(
                {line.split(":")[0] for line in subset_df.index.tolist()}
            )

        question_cols_to_remove = [
            col
            for col in df.columns
            if col.startswith("question_") and col not in question_subset
        ]
        df.drop(columns=question_cols_to_remove, inplace=True)
        logger.info(f"Using {len(question_subset)} questions from subset")

    # Set evaluation parameters
    region = getattr(config.evaluation, "region", "EU")
    model_path = getattr(
        config.evaluation,
        "gmm_model_path",
        "data/processed/gmm_models/gmm_n4_seed4242.pkl",
    )

    # Run evaluation
    logger.info(f"Evaluating {region} data...")
    results = evaluate_survey_data(df, model_path, region)

    # Print results
    print(f"\n{'=' * 50}")
    print(f"EVALUATION RESULTS FOR {region}")
    print(f"{'=' * 50}")
    print(f"Samples: {results['n_samples']:,}")
    print(f"Questions: {results['n_questions']}")
    print(f"Average log-likelihood: {results['avg_log_likelihood']:.4f}")
    print(
        f"Log-likelihood range: "
        f"[{results['sample_log_likelihoods'].min():.2f}, "
        f"{results['sample_log_likelihoods'].max():.2f}]"
    )
    print(f"Log-likelihood std: {results['sample_log_likelihoods'].std():.2f}")
    print(
        f"Probability range: "
        f"[{results['probabilities'].min():.4f}, "
        f"{results['probabilities'].max():.4f}]"
    )
    print(f"Probability mean: {results['probabilities'].mean():.4f}")
    print(f"Probability std: {results['probabilities'].std():.4f}")


if __name__ == "__main__":
    main()
