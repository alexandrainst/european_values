"""Optimisation of the questions to be used in the survey."""

import logging
import warnings

import numpy as np
import pandas as pd
import scipy.optimize as opt
from omegaconf import DictConfig
from pandas.errors import PerformanceWarning
from sklearn.metrics import davies_bouldin_score, silhouette_samples

logger = logging.getLogger(__name__)


warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=PerformanceWarning)


def optimise_survey(
    survey_df: pd.DataFrame, use_country_groups: bool, config: DictConfig
) -> pd.DataFrame:
    """Optimise the survey data by locating the most important questions.

    This optimises the separation between the country groups by maximising the
    silhouette score of the questions in the survey data.

    Args:
        survey_df:
            The survey data.
        use_country_groups:
            Whether to use country groups or individual countries for the optimisation.
        config:
            The Hydra config.

    Returns:
        A DataFrame containing the survey data with only the selected questions and
        the non-question columns.
    """
    # Get the country groupings, which depends on whether we are working with countries
    # or country groups
    country_grouping_str = "country_group" if use_country_groups else "country_code"
    unique_country_groupings = (
        survey_df.country_group.unique()
        if use_country_groups
        else survey_df.country_code.unique()
    )

    sample_df = pd.concat(
        [
            survey_df.query(f"{country_grouping_str} == @country_grouping").sample(
                n=config.sample_size_per_group, random_state=4242
            )
            for country_grouping in unique_country_groupings
        ]
    ).reset_index(drop=True)

    question_columns = [col for col in sample_df.columns if col.startswith("question_")]
    num_questions = len(question_columns)

    result = opt.differential_evolution(
        func=davies_bouldin_index,
        args=(sample_df,),
        bounds=[(0, 1)] * num_questions,
        popsize=config.population_size,
        maxiter=config.max_iterations,
        workers=config.n_jobs,
        disp=True,
        constraints=opt.LinearConstraint(
            A=np.ones((1, num_questions)), lb=config.min_questions, ub=num_questions
        ),
        integrality=np.array([True] * num_questions, dtype=bool),
        updating="deferred",
        polish=False,
    )

    identified_questions = [
        question_columns[i]
        for i, value in enumerate(np.round(result.x).astype(bool))
        if value
    ]
    logger.info(
        f"Identified {len(identified_questions):,} questions for the survey, with a "
        f"Davies-Bouldin index of {result.fun:.4f}:\n\t- "
        + "\n\t- ".join(identified_questions)
    )

    non_question_columns = [
        col for col in sample_df.columns if not col.startswith("question_")
    ]
    return survey_df.loc[:, non_question_columns + identified_questions]


def negative_silhouette_score(
    question_mask: np.ndarray, survey_df: pd.DataFrame, focus: str | None
) -> float:
    """Calculate the negative silhouette score for the given questions.

    Args:
        question_mask:
            A boolean mask indicating which questions to use, of shape (n_questions,).
        survey_df:
            The survey data, which must contain columns starting with "question_"
            and a column "country_group" indicating the country group for each row.
        focus:
            The group to focus on, where focusing here means that we only consider the
            silhouette coefficients of the rows that belong to this group. If None then
            all rows are considered.

    Returns:
        The negative silhouette score of the survey with the given questions.
    """
    # If there are no chosen questions, return 0
    if question_mask.sum().item() == 0:
        return 0

    # Ensure that the question_mask is a boolean array
    question_mask = np.round(question_mask).astype(bool)

    # Get the embedding matrix containing the question responses for the selected
    # questions
    question_columns = [col for col in survey_df.columns if col.startswith("question_")]
    embedding_matrix = survey_df[question_columns].values
    assert isinstance(embedding_matrix, np.ndarray)
    embedding_matrix = embedding_matrix[:, question_mask]

    # Compute the silhouette coefficients for either all rows or only the focus group
    silhouette_coefficients = silhouette_samples(
        X=embedding_matrix, labels=survey_df.country_group
    )
    focus_rows = (
        survey_df.query("country_group == @focus").index
        if focus is not None
        else survey_df.index
    )

    # Aggregate the silhouette scores for the focus group (or all rows)
    silhouette_score = np.mean(silhouette_coefficients[focus_rows])

    return -silhouette_score


def davies_bouldin_index(question_mask: np.ndarray, survey_df: pd.DataFrame) -> float:
    """Calculate the Davies-Bouldin index for the given questions.

    Args:
        question_mask:
            A boolean mask indicating which questions to use, of shape (n_questions,).
        survey_df:
            The survey data, which must contain columns starting with "question_"
            and a column "country_group" indicating the country group for each row.

    Returns:
        The Davies-Bouldin index of the survey with the given questions.
    """
    # If there are no chosen questions, return 100
    if question_mask.sum().item() == 0:
        return 100.0

    # Ensure that the question_mask is a boolean array
    question_mask = np.round(question_mask).astype(bool)

    # Get the embedding matrix containing the question responses for the selected
    # questions
    question_columns = [col for col in survey_df.columns if col.startswith("question_")]
    embedding_matrix = survey_df[question_columns].values
    assert isinstance(embedding_matrix, np.ndarray)
    embedding_matrix = embedding_matrix[:, question_mask]

    # Compute the Davies-Bouldin index
    return davies_bouldin_score(X=embedding_matrix, labels=survey_df.country_group)
