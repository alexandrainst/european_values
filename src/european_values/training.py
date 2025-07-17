"""Training classification models on the dataset."""

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import cross_validate
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_model(
    survey_df: pd.DataFrame,
    n_cross_val: int,
    n_jobs: int,
    n_estimators: int,
    seed: int,
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
        seed:
            The random seed for reproducibility.
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
    model = XGBClassifier(n_estimators=n_estimators, random_state=seed)

    # Train the model
    logger.info(f"Training the model with {n_cross_val}-fold cross-validation...")
    metrics = ["f1", "precision", "recall", "accuracy"]
    scores = cross_validate(
        estimator=model,
        X=embedding_matrix,
        y=labels,
        cv=n_cross_val,
        n_jobs=n_jobs,
        scoring=metrics,
    )
    mean_scores = [np.mean(scores[f"test_{metric}"]) for metric in metrics]
    std_errs = [
        np.std(scores[f"test_{metric}"], ddof=1) / np.sqrt(n_cross_val)
        for metric in metrics
    ]
    logger.info(
        "Cross-validation scores:\n"
        + "\n".join(
            f"\t{metric}: {mean:.2%} ± {1.96 * std_err:.2%} (95% CI)"
            for metric, mean, std_err in zip(metrics, mean_scores, std_errs)
        )
    )

    # Fit the model on the full dataset, as this is needed for SHAP values
    logger.info("Training the model on the full dataset to get feature importances...")
    model.fit(X=embedding_matrix, y=labels)

    # Get the most important questions
    logger.info("Calculating feature importances...")
    shortened_question_columns = []
    for question in question_columns:
        _, question_number, maybe_question_number = question.split("_", 2)
        new_question = question_number
        if re.match(r"^\d+$", maybe_question_number):
            new_question += f"_{maybe_question_number}"
        choice = question.split("choice")[-1] if "choice" in question else None
        if choice is not None:
            new_question += f":{choice}"
        shortened_question_columns.append(new_question.upper())
    explainer = shap.TreeExplainer(model=model, feature_names=question_columns)
    shap_values = explainer(
        X=embedding_matrix, check_additivity=False, approximate=fast_shap
    )
    importances = shap_values.values
    assert isinstance(importances, np.ndarray), "SHAP values should be a numpy array"
    importances = np.abs(importances).mean(axis=0)
    sorted_question_indices = np.argsort(importances)[::-1]
    logger.info(
        "Most important questions:\n"
        + "\n".join(
            f"\t{question} (importance: {importance})"
            for question, importance in zip(
                np.array(question_columns)[sorted_question_indices],
                importances[sorted_question_indices],
            )
        )
    )

    # Create a summary plot of the feature importances
    plot_path = Path("gfx", "shap_feature_importance_summary.png")
    shap.plots.beeswarm(
        shap_values=shap_values,
        max_display=100,
        group_remaining_features=False,
        show=False,
    )
    plt.title("SHAP Feature Importance Summary")
    plt.savefig(plot_path.as_posix(), bbox_inches="tight", dpi=300)
    logger.info(f"Feature importance summary plot saved as {plot_path.as_posix()!r}")
