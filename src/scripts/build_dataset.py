"""Build the dataset.

Usage:
    uv run src/scripts/build_dataset.py <config_key>=<config_value> ...
"""

import hydra
from omegaconf import DictConfig

from european_values.data_loading import load_evs_trend_data, load_evs_wvs_data


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(config: DictConfig) -> None:
    """Main function.

    Args:
        config:
            The Hydra config for your project.
    """
    load_evs_trend_data()
    load_evs_wvs_data()


if __name__ == "__main__":
    main()
