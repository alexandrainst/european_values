"""LLM evaluation using trained GMM."""

from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def load_gmm_pipeline(model_path: str) -> Pipeline:
    """Load trained GMM pipeline (scaler + model)."""
    return joblib.load(model_path)


def process_responses(responses: np.ndarray) -> np.ndarray:
    """Handle NaN values only - normalization will be done by the pipeline."""
    # Handle NaN values
    if np.isnan(responses).any():
        col_means = np.nanmean(responses, axis=0)
        # Find columns where the mean is NaN (i.e., all values are NaN)
        nan_cols = np.isnan(col_means)
        if np.any(nan_cols):
            global_mean = np.nanmean(responses)
            if np.isnan(global_mean):
                global_mean = 0.0
            col_means[nan_cols] = global_mean
        responses = np.nan_to_num(responses, nan=col_means)
    return responses


def evaluate_with_gmm(
    responses: np.ndarray, gmm_pipeline: Pipeline
) -> Tuple[float, np.ndarray]:
    """Evaluate responses using GMM pipeline."""
    # Get GMM component to use predict_proba
    gmm_model = gmm_pipeline.named_steps["gaussianmixture"]

    # Transform data through the pipeline's scaler
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
    # Load pipeline (scaler + model)
    gmm_pipeline = load_gmm_pipeline(gmm_model_path)

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

    # Process responses (handle NaN values)
    responses = process_responses(responses)

    # Evaluate
    avg_probability, sample_probabilities = evaluate_with_gmm(responses, gmm_pipeline)

    return {
        "avg_probability": avg_probability,
        "sample_probabilities": sample_probabilities,
        "n_samples": len(data),
        "n_questions": len(question_cols),
    }
