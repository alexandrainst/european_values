"""Optimisation of the questions to be used in the survey."""

import logging
import warnings
from functools import partial

import numpy as np
import pandas as pd
import scipy.optimize as opt
from omegaconf import DictConfig
from pandas.errors import PerformanceWarning
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances, silhouette_samples
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


warnings.filterwarnings(action="ignore", category=FutureWarning)
warnings.filterwarnings(action="ignore", category=UserWarning)
warnings.filterwarnings(action="ignore", category=PerformanceWarning)


def optimise_survey(survey_df: pd.DataFrame, config: DictConfig) -> pd.DataFrame:
    """Optimise the survey data by locating the most important questions.

    This optimises the separation between the country groups by maximising the
    silhouette score of the questions in the survey data.

    Args:
        survey_df:
            The survey data.
        config:
            The Hydra config.

    Returns:
        A DataFrame containing the survey data with only the selected questions and
        the non-question columns.
    """
    country_grouping_str = (
        "country_group" if config.use_country_groups else "country_code"
    )

    if config.optimisation.sample_size_per_group is None:
        sample_df = survey_df.copy()
    else:
        unique_country_groupings = (
            survey_df.country_group.unique()
            if config.use_country_groups
            else survey_df.country_code.unique()
        )
        sample_df = pd.concat(
            [
                survey_df.query(f"{country_grouping_str} == @country_grouping").sample(
                    n=config.optimisation.sample_size_per_group,
                    random_state=config.seed,
                )
                for country_grouping in unique_country_groupings
            ]
        ).reset_index(drop=True)

    question_columns = [col for col in sample_df.columns if col.startswith("question_")]
    num_questions = len(question_columns)

    match config.optimisation.method:
        case "silhouette":
            logger.info("Optimising the survey using the silhouette score...")
            func = partial(
                negative_silhouette_score,
                min_questions=config.optimisation.min_questions,
                seed=config.seed,
            )
        case "davies_bouldin":
            logger.info("Optimising the survey using the Davies-Bouldin index...")
            func = partial(
                davies_bouldin_index,
                min_questions=config.optimisation.min_questions,
                seed=config.seed,
            )
        case "centroid_distance":
            logger.info("Optimising the survey using the centroid distance...")
            func = partial(
                centroid_distance,
                min_questions=config.optimisation.min_questions,
                seed=config.seed,
            )
        case _:
            raise ValueError(
                f"Unknown optimisation method: {config.optimisation.method!r}. "
                "Must be one of 'silhouette', 'davies_bouldin', or 'centroid_distance'."
            )

    focus_log = f" with a focus on {config.focus!r}" if config.focus is not None else ""
    logger.info(
        f"Optimising {len(sample_df):,} samples for "
        f"{config.optimisation.max_iterations:,} iterations{focus_log}..."
    )
    result = opt.differential_evolution(
        func=func,
        args=(sample_df, config.focus, country_grouping_str),
        bounds=[(0, 1)] * num_questions,
        x0=np.ones(num_questions),
        popsize=config.optimisation.population_size,
        maxiter=config.optimisation.max_iterations,
        workers=config.optimisation.n_jobs,
        constraints=opt.LinearConstraint(
            A=np.ones((1, num_questions)),
            lb=config.optimisation.min_questions,
            ub=config.optimisation.max_questions or num_questions,
        ),
        integrality=np.array([True] * num_questions, dtype=bool),
        updating="deferred",
        polish=False,
        callback=callback,
        rng=config.seed,
    )

    identified_questions = [
        question_columns[i]
        for i, value in enumerate(np.round(result.x).astype(bool))
        if value
    ]
    logger.info(
        f"Identified {len(identified_questions):,} questions for the survey, with a "
        f"{config.optimisation.method} value of {result.fun:.4f}:\n\t- "
        + "\n\t- ".join(identified_questions)
    )

    non_question_columns = [
        col for col in sample_df.columns if not col.startswith("question_")
    ]
    return survey_df.loc[:, non_question_columns + identified_questions]


def negative_silhouette_score(
    question_mask: np.ndarray,
    survey_df: pd.DataFrame,
    focus: str | None,
    country_grouping_str: str,
    min_questions: int,
    seed: int,
) -> float:
    """Calculate the negative silhouette score for the given questions.

    Args:
        question_mask:
            A boolean mask indicating which questions to use, of shape (n_questions,).
        survey_df:
            The survey data, which must contain columns starting with "question_".
        focus:
            The group to focus on, where focusing here means that we only consider the
            silhouette coefficients of the rows that belong to this group. If None then
            all rows are considered.
        country_grouping_str:
            The name of the column that contains the country grouping information,
            either "country_group" or "country_code".
        min_questions:
            The minimum number of questions to select for the survey.
        seed:
            The random seed to use for reproducibility.

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

    # Use PCA
    reducer = PCA(n_components=min_questions, random_state=seed)
    embedding_matrix = reducer.fit_transform(embedding_matrix)

    # Compute the silhouette coefficients for either all rows or only the focus group
    silhouette_coefficients = silhouette_samples(
        X=embedding_matrix, labels=survey_df[country_grouping_str]
    )
    focus_rows = (
        survey_df.query(f"{country_grouping_str} == @focus").index
        if focus is not None
        else survey_df.index
    )

    # Aggregate the silhouette scores for the focus group (or all rows)
    silhouette_score = np.mean(silhouette_coefficients[focus_rows])

    return -silhouette_score


def davies_bouldin_index(
    question_mask: np.ndarray,
    survey_df: pd.DataFrame,
    focus: str | None,
    country_grouping_str: str,
    min_questions: int,
    seed: int,
) -> float:
    """Calculate the Davies-Bouldin index for the given questions.

    Args:
        question_mask:
            A boolean mask indicating which questions to use, of shape (n_questions,).
        survey_df:
            The survey data, which must contain columns starting with "question_".
        focus:
            The group to focus on, where focusing here means that we only consider the
            Davies-Bouldin index of the rows that belong to this group. If None then
            all rows are considered.
        country_grouping_str:
            The name of the column that contains the country grouping information,
            either "country_group" or "country_code".
        min_questions:
            The minimum number of questions to select for the survey.
        seed:
            The random seed to use for reproducibility.

    Returns:
        The Davies-Bouldin index of the survey with the given questions.
    """
    # If there are no chosen questions, return infinity
    if question_mask.sum().item() == 0:
        return np.inf

    # Ensure that the question_mask is a boolean array
    question_mask = np.round(question_mask).astype(bool)

    # Get the embedding matrix containing the question responses for the selected
    # questions
    question_columns = [col for col in survey_df.columns if col.startswith("question_")]
    embedding_matrix = survey_df[question_columns].values
    assert isinstance(embedding_matrix, np.ndarray)
    embedding_matrix = embedding_matrix[:, question_mask]

    # Use PCA
    reducer = PCA(n_components=min_questions, random_state=seed)
    embedding_matrix = reducer.fit_transform(embedding_matrix)

    num_questions = embedding_matrix.shape[1]

    # Encode the country groups
    le = LabelEncoder()
    labels = le.fit_transform(survey_df[country_grouping_str])
    num_labels = survey_df[country_grouping_str].nunique()
    assert isinstance(num_labels, int)

    # Get the index of the focus group if it is specified
    focus_label = le.transform([focus])[0] if focus is not None else None

    # Compute the intra-cluster distances and centroids for each country group. If we
    # are focusing on a specific group, we only compute the intra-cluster distances
    # for that group.
    intra_dists: np.ndarray = np.zeros(num_labels) if focus is None else np.zeros(1)
    centroids = np.zeros((num_labels, num_questions), dtype=float)
    for k in range(num_labels):
        cluster_k = embedding_matrix[labels == k]
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid
        if focus is None:
            intra_dists[k] = np.average(pairwise_distances(cluster_k, [centroid]))
        elif k == focus_label:
            intra_dists[0] = np.average(pairwise_distances(cluster_k, [centroid]))

    # Compute the distances between centroids. If we are focusing on a specific
    # group, we only compute the distances between the centroid of that group and all
    # other centroids.
    # Shape: (num_labels, num_labels)
    if focus is None:
        centroid_distances = pairwise_distances(centroids)
    else:
        centroid_distances = pairwise_distances(centroids[focus_label, None], centroids)

    # Since we are also comparing each centroid to itself, we set those distances to
    # infinity to avoid division by zero in the next step
    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0
    centroid_distances[centroid_distances == 0] = np.inf

    # Compute the combined intra-cluster distances, where entry (i, j) is the sum of
    # the intra-cluster distances of cluster i and cluster j.
    # Shape: (num_labels, num_labels)
    if focus is None:
        intra_dists = intra_dists[:, None] + intra_dists

    # Compute the Davies-Bouldin index as the maximum of the ratio of combined intra-
    # cluster distances to centroid distances for each cluster
    scores = np.max(intra_dists / centroid_distances, axis=-1)
    return float(np.mean(scores))


def centroid_distance(
    question_mask: np.ndarray,
    survey_df: pd.DataFrame,
    focus: str | None,
    country_grouping_str: str,
    min_questions: int,
    seed: int,
) -> float:
    """Calculate the centroid distance for the given questions.

    Args:
        question_mask:
            A boolean mask indicating which questions to use, of shape (n_questions,).
        survey_df:
            The survey data, which must contain columns starting with "question_".
        focus:
            The group to focus on, where focusing here means that we only consider the
            centroid distance of the rows that belong to this group. If None then all
            rows are considered.
        country_grouping_str:
            The name of the column that contains the country grouping information,
            either "country_group" or "country_code".
        min_questions:
            The minimum number of questions to select for the survey.
        seed:
            The random seed to use for reproducibility.

    Returns:
        The centroid distance of the survey with the given questions.
    """
    # If there are no chosen questions, return infinity
    if question_mask.sum().item() == 0:
        return np.inf

    # Ensure that the question_mask is a boolean array
    question_mask = np.round(question_mask).astype(bool)

    # Get the embedding matrix containing the question responses for the selected
    # questions
    question_columns = [col for col in survey_df.columns if col.startswith("question_")]
    embedding_matrix = survey_df[question_columns].values
    assert isinstance(embedding_matrix, np.ndarray)
    embedding_matrix = embedding_matrix[:, question_mask]

    # TODO: Check if this is necessary to avoid bias towards fewer questions
    # Use PCA
    # reducer = PCA(n_components=min_questions, random_state=seed)
    # embedding_matrix = reducer.fit_transform(embedding_matrix)

    # Encode the country groups
    le = LabelEncoder()
    labels = le.fit_transform(survey_df[country_grouping_str])
    num_labels = survey_df[country_grouping_str].nunique()
    assert isinstance(num_labels, int)

    # Compute the centroids for each country group
    centroids = np.zeros((num_labels, embedding_matrix.shape[1]), dtype=float)
    for k in range(num_labels):
        cluster_k = embedding_matrix[labels == k]
        centroid = cluster_k.mean(axis=0)
        centroids[k] = centroid

    # Compute the distances between centroids
    # Shape: (num_labels, num_labels)
    centroid_distances = pairwise_distances(X=centroids, metric="cosine")

    # Since we are also comparing each centroid to itself, we set those distances to
    # infinity to avoid division by zero in the next step
    if np.allclose(centroid_distances, 0):
        return 0.0
    centroid_distances[centroid_distances == 0] = np.inf

    # If we are focusing on a specific group, we only consider the rows that belong
    # to that group
    if focus is not None:
        focus_label = le.transform([focus])[0]
        centroid_distances = centroid_distances[focus_label, :]

    # Return the mean distance between centroids
    return float(centroid_distances[centroid_distances != np.inf].mean())


def callback(intermediate_result: opt.OptimizeResult) -> None:
    """Callback function to log the intermediate results of the optimisation.

    Args:
        intermediate_result:
            The intermediate result of the optimisation.
    """
    logger.info(
        f"Iteration {intermediate_result.nit:,}: "
        f"Objective function value: {intermediate_result.fun:.4f}, "
        f"number of questions: {intermediate_result.x.sum():,}"
    )
