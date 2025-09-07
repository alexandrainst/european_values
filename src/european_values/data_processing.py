"""Data processing functions."""

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from feature_engine.imputation import RandomSampleImputer
from omegaconf import DictConfig
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from .utils import df_has_column_with_only_nans, group_country

logger = logging.getLogger(__name__)


def process_data(
    df: pd.DataFrame, config: DictConfig, normalize: bool = True
) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Process the survey data.

    Args:
        df:
            The survey data.
        config:
            The Hydra config.
        normalize (optional):
            Whether to apply normalization. If False, scaler is fitted but not applied.
            Defaults to True.

    Returns:
        The processed DataFrame and the fitted scaler.

    Raises:
        ValueError:
            If any country or country group in the config is not present in the data.
    """
    # Group countries
    logger.info("Grouping countries into regions...")
    df["country_group"] = df.country_code.apply(group_country)

    # Filter countries
    if config.countries is not None:
        missing_countries = set(config.countries) - set(df.country_code.unique())
        if missing_countries:
            logger.warning(
                f"The countries {missing_countries} in the config are not present "
                "in the data. Ignoring them."
            )
        logger.info(f"Filtering data for countries: {config.countries}")
        df = df.query("country_code in @config.countries").reset_index(drop=True)
        logger.info(f"Shape of the data after filtering: {df.shape}")

    # Filter country groups
    if config.country_groups is not None:
        missing_groups = set(config.country_groups) - set(df.country_group.unique())
        if missing_groups:
            logger.warning(
                f"The country groups {missing_groups} in the config are not present "
                "in the data. Ignoring them."
            )
        logger.info(f"Filtering data for country groups: {config.country_groups}")
        df = df.query("country_group in @config.country_groups").reset_index(drop=True)
        logger.info(f"Shape of the data after filtering: {df.shape}")

    # Get the country groupings, which depends on whether we are working with countries
    # or country groups
    country_grouping_str = (
        "country_group" if config.use_country_groups else "country_code"
    )
    unique_country_groupings = (
        df.country_group.unique()
        if config.use_country_groups
        else df.country_code.unique()
    )

    # Remove questions for which a country or country group exists that have not
    # answered the question
    questions_with_missing_answers: dict[str, set[str]] = defaultdict(set)
    for country_grouping in unique_country_groupings:
        country_grouping_df = df.query(f"{country_grouping_str} == @country_grouping")
        na_df = country_grouping_df.isna().all(axis=0)
        assert isinstance(na_df, pd.Series)
        na_df = na_df[na_df]
        assert isinstance(na_df, pd.Series)
        questions = [
            question
            for question in na_df.index.tolist()
            if str(question).startswith("question_")
        ]
        for question in questions:
            assert isinstance(question, str)
            questions_with_missing_answers[question].add(country_grouping)
    if questions_with_missing_answers:
        df = df.drop(columns=list(questions_with_missing_answers.keys()))
        questions_removed_str = "\n\t- ".join(
            [
                f"{question} (missing for {', '.join(groupings)})"
                for question, groupings in questions_with_missing_answers.items()
            ]
        )
        num_questions_after_removal = len(
            [col for col in df.columns if col.startswith("question_")]
        )
        logger.info(
            f"Removed {len(questions_with_missing_answers)} questions where at least "
            f"one country group has not answered:\n\t- {questions_removed_str}"
        )
        logger.info(
            f"Shape of the data after removing questions with missing answers: "
            f"{df.shape}, where {num_questions_after_removal:,} are question columns."
        )

    # Count the number of missing values in the question columns
    question_columns = [col for col in df.columns if col.startswith("question_")]
    num_missing = df[question_columns].isna().sum().sum()
    num_total = df[question_columns].size
    pct_missing = num_missing / num_total

    # Initialise the imputer
    sampler = RandomSampleImputer(random_state=4242, variables=question_columns)

    # Impute missing values
    embedding_matrix = np.empty(shape=(df.shape[0], len(question_columns)))
    for country_grouping in tqdm(
        iterable=unique_country_groupings,
        desc=f"Imputing {num_missing:,} ({pct_missing:.2%}) missing values...",
        unit=country_grouping_str,
    ):
        country_grouping_df = df.query(f"{country_grouping_str} == @country_grouping")

        # Get all combinations of metadata values, as we want to impute values
        # conditioned on these demographic features
        metadata_combinations = list(
            {
                (row.education, row.sex, row.age_interval)
                for _, row in country_grouping_df.iterrows()
            }
        )

        # Iterate over all combinations of metadata values
        for education, sex, age_interval in tqdm(
            iterable=metadata_combinations,
            desc=f"Imputing values for {country_grouping!r} conditioned on "
            "metadata values",
            leave=False,
            unit="metadata combination",
        ):
            # Slice the DataFrame to contain exactly the examples we want to impute
            df_to_be_imputed = country_grouping_df.copy()
            if pd.isna(education):
                df_to_be_imputed = df_to_be_imputed.query("education.isna()")
            else:
                df_to_be_imputed = df_to_be_imputed.query("education == @education")
            if pd.isna(sex):
                df_to_be_imputed = df_to_be_imputed.query("sex.isna()")
            else:
                df_to_be_imputed = df_to_be_imputed.query("sex == @sex")
            if pd.isna(age_interval):
                df_to_be_imputed = df_to_be_imputed.query("age_interval.isna()")
            else:
                df_to_be_imputed = df_to_be_imputed.query(
                    "age_interval == @age_interval"
                )

            assert not df_to_be_imputed.empty, (
                f"DataFrame to be imputed for {country_grouping!r} with "
                f"education={education}, sex={sex}, and "
                f"age_interval={age_interval} is empty. This should not happen."
            )

            # If we do not have any rows to impute, we just use the original
            # values
            if df_to_be_imputed[question_columns].isna().sum().sum() == 0:
                imputed_values = df_to_be_imputed[question_columns].values
                embedding_matrix[df_to_be_imputed.index, :] = imputed_values
                continue

            # Slice the DataFrame to get the pool of values we can sample from to
            # impute. We only slice if the metadata values are not NaN and if the
            # resulting sliced DataFrame has more than one row.
            imputation_values_df = country_grouping_df.copy()
            if not pd.isna(education) and not df_has_column_with_only_nans(
                df=imputation_values_df
            ):
                candidate_imputation_values_df = imputation_values_df.query(
                    "education == @education"
                )
                if not df_has_column_with_only_nans(df=candidate_imputation_values_df):
                    imputation_values_df = candidate_imputation_values_df
            if not pd.isna(sex) and not df_has_column_with_only_nans(
                df=imputation_values_df
            ):
                candidate_imputation_values_df = imputation_values_df.query(
                    "sex == @sex"
                )
                if not df_has_column_with_only_nans(df=candidate_imputation_values_df):
                    imputation_values_df = candidate_imputation_values_df
            if not pd.isna(age_interval) and not df_has_column_with_only_nans(
                df=imputation_values_df
            ):
                candidate_imputation_values_df = imputation_values_df.query(
                    "age_interval == @age_interval"
                )
                if not df_has_column_with_only_nans(df=candidate_imputation_values_df):
                    imputation_values_df = candidate_imputation_values_df

            # Impute the missing values for each question column
            sampler.fit(X=imputation_values_df)
            imputed_values = sampler.transform(X=df_to_be_imputed)[
                question_columns
            ].values
            embedding_matrix[df_to_be_imputed.index, :] = imputed_values

    # Check if we have imputed all missing values
    assert np.isnan(embedding_matrix).sum() == 0, (
        f"Not all missing values have been imputed. Found "
        f"{np.isnan(embedding_matrix).sum():,} missing values in the embedding matrix."
    )

    # Always fit the scaler (so we can save it), but only apply if requested
    logger.info("Fitting scaler...")
    scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    scaler.fit(embedding_matrix)

    if normalize:
        logger.info("Applying normalization...")
        embedding_matrix = scaler.transform(embedding_matrix)
    else:
        logger.info("Skipping normalization (but scaler is fitted and available)...")

    # Update the survey DataFrame with the processed values
    df[question_columns] = embedding_matrix

    return df, scaler
