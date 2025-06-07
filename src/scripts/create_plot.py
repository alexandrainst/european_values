"""Create plots with the data.

Usage:
    uv run src/scripts/create_plot.py <config_key>=<config_value> ...
"""

import logging

import hydra
import pandas as pd
from datasets import Dataset, load_dataset
from omegaconf import DictConfig

from european_values.data_processing import process_data
from european_values.plotting import create_scatter

logger = logging.getLogger("create_plot")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main function.

    Args:
        config:
            The Hydra config for your project.
    """
    dataset_id_suffix = "_processed" if config.countries == "all" else ""
    match config.data:
        case "evs_trend":
            logger.info("Using EVS trend data.")
            dataset = load_dataset(
                path=config.repo_id,
                name="evs_trend_data_1981_2017" + dataset_id_suffix,
                split="train",
            )
            assert isinstance(dataset, Dataset)
            df = dataset.to_pandas()
            dataset_name = "EVS trend data"
        case "evs_wvs":
            logger.info("Using EVS/WVS data.")
            dataset = load_dataset(
                path=config.repo_id,
                name="evs_wvs_data_2017_2022" + dataset_id_suffix,
                split="train",
            )
            assert isinstance(dataset, Dataset)
            df = dataset.to_pandas()
            dataset_name = "EVS/WVS data"
        case None:
            raise ValueError(
                "No data selected. Please set `data` to be either 'evs_trend' or "
                "'evs_wvs'."
            )
        case _:
            raise ValueError(
                f"Invalid data selected: {config.data}. Please set `data` to be either "
                "'evs_trend' or 'evs_wvs'."
            )
    assert isinstance(df, pd.DataFrame)

    if config.countries != "all":
        logger.info(f"Filtering data for countries: {config.countries}")
        df = df.query("country_code in @config.countries").reset_index(drop=True)
        df = process_data(df=df, imputation_neighbours=config.imputation_neighbours)
        logger.info(f"Shape of the data after filtering: {df.shape}")

    create_scatter(survey_df=df, dataset_name=dataset_name, config=config)


if __name__ == "__main__":
    main()
