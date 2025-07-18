"""Training classification models on the dataset."""

import logging
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from shap import kmeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def train_model(
    survey_df: pd.DataFrame,
    model_type: Literal[
        "xgboost",
        "logistic_regression",
        "support_vector_machine",
        "naive_bayes_gaussian",
        "naive_bayes_multinomial",
    ],
    n_cross_val: int,
    n_jobs: int,
    n_estimators: int,
    seed: int,
    bootstrap: bool,
) -> None:
    """Train a classifier that classifies survey data into country groups.

    Args:
        survey_df:
            The DataFrame containing the survey data. It should have the features
            in the columns and the target variable (country group) in the last column.
        model_type:
            The type of model to train.
        n_cross_val:
            The number of cross-validation folds to use.
        n_jobs:
            The number of jobs to run in parallel for cross-validation. If -1, all CPUs
            are used.
        n_estimators:
            The number of trees in the random forest.
        seed:
            The random seed to use for reproducibility.
        bootstrap:
            Whether to bootstrap the data for training, using the `seed` parameter.

    Raises:
        ValueError:
            If an unsupported model type is provided.
    """
    logging.getLogger("shap").setLevel(logging.CRITICAL)

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

    # Bootstrap the embedding matrix and the labels
    if bootstrap:
        logger.info(
            f"Bootstrapping the embedding matrix and labels with seed {seed}..."
        )
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            a=np.arange(len(embedding_matrix)), size=len(embedding_matrix), replace=True
        )
        embedding_matrix = embedding_matrix[indices]
        labels = np.array(labels)[indices]

    # Load the model
    match model_type:
        case "xgboost":
            model = XGBClassifier(n_estimators=n_estimators)
        case "logistic_regression":
            model = LogisticRegression()
        case "support_vector_machine":
            model = SVC(probability=True, random_state=seed)
        case "naive_bayes_gaussian":
            model = GaussianNB()
        case "naive_bayes_multinomial":
            model = MultinomialNB()
        case _:
            raise ValueError(
                "Unsupported model type: {model_type}. Supported types are: "
                "'xgboost', 'logistic_regression', 'support_vector_machine', "
                "'naive_bayes_gaussian', 'naive_bayes_multinomial'. Please check "
                "your configuration."
            )

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

    # Get the SHAP explainer
    match model_type:
        case "xgboost" | "random_forest":
            logger.info("Using the TreeExplainer for SHAP values.")
            explainer = shap.TreeExplainer(model=model, feature_names=question_columns)
        case "logistic_regression":
            logger.info("Using the LinearExplainer for SHAP values.")
            explainer = shap.LinearExplainer(
                model=model, masker=embedding_matrix, feature_names=question_columns
            )
        case _:
            logger.info(
                "Using the KernelExplainer for SHAP values. Clustering the data "
                "into a smaller number of clusters to speed up the computation..."
            )
            clustered_data = kmeans(X=embedding_matrix, k=10)
            explainer = shap.KernelExplainer(
                model=model.predict_proba,
                data=clustered_data,
                feature_names=question_columns,
            )

    # Get the most important questions
    logger.info("Calculating feature importances...")
    shap_values = explainer(embedding_matrix)
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

    # Set up paths
    questions_path = Path("data", f"important_questions_{model_type}_seed{seed}.csv")
    plot_path = Path(
        "gfx", f"shap_feature_importance_summary_{model_type}_seed{seed}.png"
    )
    version = 1
    while questions_path.exists() or plot_path.exists():
        version += 1
        questions_path = questions_path.with_name(
            questions_path.stem + f"_v{version}" + questions_path.suffix
        )
        plot_path = plot_path.with_name(
            plot_path.stem + f"_v{version}" + plot_path.suffix
        )

    # Save the important questions to a CSV file
    pd.DataFrame(
        {
            "model_type": model_type,
            "seed": seed,
            "question": np.array(question_columns)[sorted_question_indices],
            "importance": importances[sorted_question_indices],
        }
    ).to_csv(questions_path.as_posix(), index=False, encoding="utf-8")
    logger.info(f"Important questions saved to {questions_path.as_posix()!r}")

    # Create a summary plot of the feature importances
    shap.plots.beeswarm(
        shap_values=shap_values,
        max_display=100,
        group_remaining_features=False,
        show=False,
    )
    plt.title("SHAP Feature Importance Summary")
    plt.savefig(plot_path.as_posix(), bbox_inches="tight", dpi=300)
    plt.close()
    logger.info(f"Feature importance summary plot saved as {plot_path.as_posix()!r}")
