"""Train a typicality-based pipeline (KDE/ECDF or kNN/ECDF)."""

import logging

import hydra
import pandas as pd
from omegaconf import DictConfig

from european_values.data_loading import load_evs_trend_data, load_evs_wvs_data
from european_values.data_processing import process_data
from european_values.typicality_training import (
    train_kde_ecdf_pipeline,
    train_knn_ecdf_pipeline,
)
from european_values.utils import apply_subset_filtering

logger = logging.getLogger("train_typicality_pipeline")


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

    eu_df = df.query("country_group == 'EU'")
    method = config.typicality.method.lower()

    if method == "kde":
        train_kde_ecdf_pipeline(
            eu_df=eu_df,
            scaler=scaler,
            bandwidth=config.typicality.kde_bandwidth,
            output_path=config.typicality.model_path,
        )
    elif method == "knn":
        train_knn_ecdf_pipeline(
            eu_df=eu_df,
            scaler=scaler,
            k=config.typicality.knn_k,
            metric=config.typicality.knn_metric,
            n_jobs=config.typicality.knn_n_jobs,
            output_path=config.typicality.model_path,
        )
    else:
        raise ValueError(
            f"Unknown typicality method {config.typicality.method!r}. "
            "Use 'kde' or 'knn'."
        )


if __name__ == "__main__":
    main()
