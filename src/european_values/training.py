"""Training classification models on the dataset."""

import logging

import numpy as np
import pandas as pd
from shap import TreeExplainer
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_model(
    survey_df: pd.DataFrame,
    n_cross_val: int,
    n_jobs: int,
    n_estimators: int,
    fast_shap: bool,
) -> None:
    """Train a random forest classifier that classifies survey data into country groups.

    Args:
        survey_df:
            The DataFrame containing the survey data. It should have the features
            in the columns and the target variable (country group) in the last column.
        n_cross_val:
            The number of cross-validation folds to use.
        n_jobs:
            The number of jobs to run in parallel for cross-validation. If -1, all
            CPUs are used.
        n_estimators:
            The number of trees in the random forest.
        fast_shap:
            Whether the SHAP values should be computed using the fast method, which is
            less accurate but faster.

    Raises:
        ValueError:
            If an unsupported model type is provided.
    """
    # Create the embedding matrix
    logger.info("Creating embedding matrix...")
    question_columns = [col for col in survey_df.columns if col.startswith("question_")]
    embedding_matrix = survey_df[question_columns].values

    # Set up the labels
    logger.info("Setting up binary labels...")
    labels = [
        1 if country_group == "Europe" else 0
        for country_group in survey_df.country_group
    ]

    # Load the model
    model = XGBClassifier(n_estimators=n_estimators, random_state=8446)

    # Train the model
    logger.info(f"Training the model with {n_cross_val}-fold cross-validation...")
    scores = cross_validate(
        estimator=model, X=embedding_matrix, y=labels, cv=n_cross_val, n_jobs=n_jobs
    )
    mean_score = scores["test_score"].mean()
    std_err = np.std(scores["test_score"], ddof=1) / np.sqrt(n_cross_val)
    logger.info(
        f"Model trained with mean accuracy: {mean_score:.4f} ± {1.96 * std_err:.4f} "
        f"(95% confidence interval, {n_cross_val}-fold cross-validation)"
    )

    # Fit the model on the full dataset, as this is needed for SHAP values
    logger.info("Training the model on the full dataset to get feature importances...")
    model.fit(X=embedding_matrix, y=labels)

    # Get the most important questions
    logger.info("Calculating feature importances...")
    explainer = TreeExplainer(model=model, feature_names=question_columns)
    importances = explainer(
        X=embedding_matrix, check_additivity=False, approximate=fast_shap
    ).values
    assert isinstance(importances, np.ndarray), "SHAP values should be a numpy array"
    importances = importances.mean(axis=0)
    sorted_question_indices = np.argsort(importances)[::-1]
    logger.info(
        "Most important questions:\n"
        + "\n".join(
            f"\t{question} (importance: {importance})"
            for question, importance in zip(
                np.array(question_columns)[sorted_question_indices],
                importances[sorted_question_indices],
            )
            if importance > 0
        )
    )
