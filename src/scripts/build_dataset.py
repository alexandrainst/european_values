"""Build the dataset.

Usage:
    uv run src/scripts/build_dataset.py <config_key>=<config_value> ...
"""

import logging

import hydra
from datasets import Dataset
from omegaconf import DictConfig

from european_values.data_loading import load_evs_trend_data, load_evs_wvs_data
from european_values.data_processing import process_data

logger = logging.getLogger("build_dataset")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main function.

    Args:
        config:
            The Hydra config for your project.
    """
    # Load and push the EVS trend data to the Hugging Face Hub
    evs_trend_df = load_evs_trend_data()
    Dataset.from_pandas(evs_trend_df, preserve_index=False).push_to_hub(
        repo_id=config.repo_id, config_name="evs_trend_data_1981_2017", private=True
    )

    # Process and push the EVS trend data to the Hugging Face Hub
    processed_evs_trend_df = process_data(
        df=evs_trend_df, imputation_neighbours=config.imputation_neighbours
    )
    Dataset.from_pandas(processed_evs_trend_df, preserve_index=False).push_to_hub(
        repo_id=config.repo_id,
        config_name="evs_trend_data_1981_2017_processed",
        private=True,
    )

    # Load and push the EVS/WVS data to the Hugging Face Hub
    evs_wvs_df = load_evs_wvs_data()
    Dataset.from_pandas(evs_wvs_df, preserve_index=False).push_to_hub(
        repo_id=config.repo_id, config_name="evs_wvs_data_2017_2022", private=True
    )

    # Process and push the EVS/WVS data to the Hugging Face Hub
    processed_evs_wvs_df = process_data(
        df=evs_wvs_df, imputation_neighbours=config.imputation_neighbours
    )
    Dataset.from_pandas(processed_evs_wvs_df, preserve_index=False).push_to_hub(
        repo_id=config.repo_id,
        config_name="evs_wvs_data_2017_2022_processed",
        private=True,
    )


if __name__ == "__main__":
    main()
