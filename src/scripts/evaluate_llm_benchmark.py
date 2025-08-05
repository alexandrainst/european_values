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
    # Load data
    logger.info("Loading data...")
    df = load_evs_wvs_data()

    # Process data but SKIP normalization (let pipeline handle it)
    df, _ = process_data(df=df, config=config, normalize=False)  # Fixed!

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

    # Set evaluation parameters
    region = getattr(config.evaluation, "region", "EU")
    model_path = getattr(
        config.evaluation,
        "gmm_model_path",
        "data/processed/gmm_model/gmm_n4_seed4242.pkl",
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
    print(f"Average probability: {results['avg_probability']:.4f}")
    print(
        f"Probability range: [{results['sample_probabilities'].min():.4f}, "
        f"{results['sample_probabilities'].max():.4f}]"
    )

    print(f"Probability mean: {results['sample_probabilities'].mean():.4f}")
    print(f"Probability std: {results['sample_probabilities'].std():.4f}")


if __name__ == "__main__":
    main()
