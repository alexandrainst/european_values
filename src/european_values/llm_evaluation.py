"""LLM evaluation using trained GMM."""

from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def load_gmm(model_path: str) -> GaussianMixture:
    """Load trained GMM model."""
    return joblib.load(model_path)


def process_responses(responses: np.ndarray) -> np.ndarray:
    """Handle NaN values only - no normalization."""
    # Handle NaN values
    if np.isnan(responses).any():
        responses = np.nan_to_num(responses, nan=np.nanmean(responses, axis=0))
    return responses


def evaluate_with_gmm(
    responses: np.ndarray, gmm_model: GaussianMixture
) -> Tuple[float, np.ndarray]:
    """Evaluate responses using GMM."""
    avg_log_likelihood = gmm_model.score(responses)
    per_sample_log_likelihood = gmm_model.score_samples(responses)
    return avg_log_likelihood, per_sample_log_likelihood


def log_likelihood_to_probability(log_likelihood: np.ndarray) -> np.ndarray:
    """Convert log-likelihood to actual probability."""
    return np.exp(log_likelihood)


def evaluate_survey_data(
    survey_df: pd.DataFrame, gmm_model_path: str, region: str = "EU"
) -> Dict[str, Any]:
    """Evaluate survey data with GMM."""
    # Load model
    gmm = load_gmm(gmm_model_path)

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

    # Evaluate
    avg_score, sample_scores = evaluate_with_gmm(responses, gmm)
    probabilities = log_likelihood_to_probability(sample_scores)

    return {
        "avg_log_likelihood": avg_score,
        "sample_log_likelihoods": sample_scores,
        "probabilities": probabilities,
        "n_samples": len(data),
        "n_questions": len(question_cols),
    }
