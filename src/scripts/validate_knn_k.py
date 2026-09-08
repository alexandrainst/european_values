"""Validate k choices for kNN typicality."""

import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import ks_2samp

from european_values.data_loading import load_evs_trend_data, load_evs_wvs_data
from european_values.data_processing import process_data
from european_values.typicality_transformers import (
    ECDFTransformer,
    KNNDistanceTransformer,
)
from european_values.utils import apply_subset_filtering

logger = logging.getLogger("validate_knn_k")


def _spearman(a: pd.Series, b: pd.Series) -> float:
    return a.corr(b, method="spearman")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main function."""
    match (config.include_evs_trend, config.include_evs_wvs):
        case (True, True):
            logger.info("Loading EVS trend and EVS/WVS data...")
            evs_trend_df = load_evs_trend_data()
            evs_wvs_df = load_evs_wvs_data()
            df = pd.concat([evs_trend_df, evs_wvs_df], ignore_index=True)
        case (True, False):
            logger.info("Loading only EVS trend data...")
            df = load_evs_trend_data()
        case (False, True):
            logger.info("Loading only EVS/WVS data...")
            df = load_evs_wvs_data()
        case _:
            raise ValueError(
                "At least one of `include_evs_trend` or `include_evs_wvs` must be True."
            )

    df = apply_subset_filtering(df=df, subset_csv_path=config.subset_csv)

    logger.info("Processing the data WITHOUT normalization...")
    df, scaler = process_data(df=df, config=config, normalize=False)

    question_cols = [col for col in df.columns if col.startswith("question_")]
    all_scaled = scaler.transform(df[question_cols].values)
    eu_mask = df["country_group"] == "EU"
    eu_scaled = all_scaled[eu_mask]

    group_col = "country_group" if config.use_country_groups else "country_code"
    ks = list(config.typicality_validation.ks)
    if len(ks) == 0:
        raise ValueError("typicality_validation.ks must contain at least one k value.")

    output_dir = Path(config.typicality_validation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_k = ks[0]
    baseline_scores = None
    baseline_group_means = None

    results = []
    group_means_by_k = {}

    for k in ks:
        logger.info(f"Evaluating k={k}...")
        knn = KNNDistanceTransformer(
            k=int(k),
            metric=config.typicality.knn_metric,
            n_jobs=config.typicality.knn_n_jobs,
        )
        eu_distances = knn.fit_transform(eu_scaled)
        ecdf = ECDFTransformer(higher_is_better=False).fit(eu_distances)

        all_distances = knn.transform(all_scaled)
        all_scores = ecdf.transform(all_distances)

        df_scores = df[[group_col]].copy()
        df_scores["score"] = all_scores

        group_means = df_scores.groupby(group_col)["score"].mean().sort_index()
        group_means_by_k[int(k)] = group_means

        if baseline_scores is None:
            baseline_scores = all_scores
            baseline_group_means = group_means
            ks_stat = 0.0
            ks_p = 1.0
            mad = 0.0
            spearman = 1.0
        else:
            ks_stat, ks_p = ks_2samp(baseline_scores, all_scores)
            mad = float(np.mean(np.abs(all_scores - baseline_scores)))
            spearman = float(_spearman(baseline_group_means, group_means))

        results.append(
            dict(
                k=int(k),
                ks_statistic=ks_stat,
                ks_pvalue=ks_p,
                mean_abs_diff=mad,
                group_mean_spearman=spearman,
            )
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "knn_k_stability.csv", index=False)

    group_means_df = pd.DataFrame(group_means_by_k)
    group_means_df.to_csv(output_dir / f"group_means_by_k_{group_col}.csv")

    logger.info(f"Stability results saved to {output_dir.as_posix()}")


if __name__ == "__main__":
    main()
