"""Optimisation of the questions to be used in the survey."""

import logging
import warnings

import numpy as np
import pandas as pd
import scipy.optimize as opt
from omegaconf import DictConfig
from pandas.errors import PerformanceWarning
from sklearn.metrics import silhouette_samples

logger = logging.getLogger(__name__)


warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=PerformanceWarning)


def optimise_survey(survey_df: pd.DataFrame, config: DictConfig) -> None:
    """Optimise the survey data by locating the most important questions.

    This optimises the separation between the country groups by maximising the
    silhouette score of the questions in the survey data.

    Args:
        survey_df:
            The survey data.
        config:
            The Hydra config.
    """
    if config.sample_size_per_group is not None:
        logger.info(
            f"Sampling {config.sample_size_per_group:,} rows from each country group "
            "to speed up the optimisation."
        )
        survey_df = pd.concat(
            [
                survey_df.query("country_group == @country_group").sample(
                    n=config.sample_size_per_group, random_state=4242
                )
                for country_group in survey_df.country_group.unique()
            ]
        ).reset_index(drop=True)

    logger.info(f"Initiating optimisation for {config.max_iterations:,} iterations...")
    num_questions = len(
        [col for col in survey_df.columns if col.startswith("question_")]
    )
    result = opt.differential_evolution(
        func=negative_silhouette_score,
        args=(survey_df, config.focus),
        bounds=[(0, 1)] * num_questions,
        x0=np.ones(num_questions),
        popsize=config.population_size,
        maxiter=config.max_iterations,
        workers=-1,
        disp=True,
        constraints=opt.LinearConstraint(
            A=np.ones((1, num_questions)), lb=config.min_questions, ub=num_questions
        ),
        integrality=np.array([True] * num_questions, dtype=bool),
    )

    question_columns = [col for col in survey_df.columns if col.startswith("question_")]
    identified_questions = [
        question_columns[i]
        for i, value in enumerate(np.round(result.x).astype(bool))
        if value
    ]
    logger.info(
        "Identified questions for the survey, with a silhouette score of "
        f"{-result.fun:.4f}:\n\t- " + "\n\t- ".join(identified_questions)
    )
    logger.info(
        "The questions that got removed from the survey:\n\t- "
        + "\n\t- ".join(
            [
                question
                for question in question_columns
                if question not in identified_questions
            ]
        )
    )


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
    question_mask = np.round(question_mask).astype(bool)
    question_columns = [col for col in survey_df.columns if col.startswith("question_")]
    embedding_matrix = survey_df[question_columns].values
    assert isinstance(embedding_matrix, np.ndarray)
    embedding_matrix = embedding_matrix[:, question_mask]

    silhouette_coefficients = silhouette_samples(
        X=embedding_matrix, labels=survey_df.country_group
    )
    focus_rows = (
        survey_df.query("country_group == @focus").index
        if focus is not None
        else survey_df.index
    )
    silhouette_score = np.mean(silhouette_coefficients[focus_rows])

    return -silhouette_score
