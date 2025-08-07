"""LLM evaluation using trained GMM."""

from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def evaluate_with_gmm(
    responses: np.ndarray, gmm_pipeline: Pipeline
) -> Tuple[float, np.ndarray]:
    """Evaluate responses using GMM pipeline."""
    # Try using the pipeline directly first
    try:
        full_probabilities = gmm_pipeline.predict_proba(responses)
    except AttributeError:
        # Fallback: Pipeline doesn't have predict_proba, access GMM component
        gmm_model = gmm_pipeline.named_steps["gaussianmixture"]
        scaled_responses = gmm_pipeline.named_steps["minmaxscaler"].transform(responses)
        full_probabilities = gmm_model.predict_proba(scaled_responses)

    print("First 5 rows of full probabilities:")
    print(full_probabilities[:5])

    probabilities = full_probabilities.max(axis=1)
    avg_probability = np.mean(probabilities)
    return avg_probability, probabilities


def evaluate_survey_data(
    survey_df: pd.DataFrame, gmm_model_path: str, region: str = "EU"
) -> Dict[str, Any]:
    """Evaluate survey data with GMM."""
    # Load pipeline directly (no need for wrapper function)
    gmm_pipeline = joblib.load(gmm_model_path)

    # Try filtering by country_group first, then by country_code
    data = survey_df[survey_df["country_group"] == region]
    if len(data) == 0:
        data = survey_df[survey_df["country_code"] == region]
    if len(data) == 0:
        available_groups = survey_df["country_group"].unique()
        available_countries = survey_df["country_code"].unique()
        raise ValueError(
            f"No data found for region '{region}'. "
            f"Available groups: {list(available_groups)}, "
            f"Available countries: {list(available_countries)}"
        )

    question_cols = [col for col in data.columns if col.startswith("question_")]
    responses = data[question_cols].values

    # Since data should already be processed and imputed, we shouldn't need
    # additional NaN handling. But keep minimal safety check:
    if np.isnan(responses).any():
        nan_count = np.isnan(responses).sum()
        print(f"Warning: Found {nan_count} NaN values in processed data")
        # Simple fallback: replace NaN with column means
        col_means = np.nanmean(responses, axis=0)
        nan_mask = np.isnan(responses)
        responses = np.where(nan_mask, col_means, responses)

    # Evaluate
    avg_probability, sample_probabilities = evaluate_with_gmm(responses, gmm_pipeline)

    return {
        "avg_probability": avg_probability,
        "sample_probabilities": sample_probabilities,
        "n_samples": len(data),
        "n_questions": len(question_cols),
    }
