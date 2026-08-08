"""Create diagnostic plots for typicality pipelines."""

import logging
from pathlib import Path

import hydra
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from european_values.data_loading import load_evs_trend_data, load_evs_wvs_data
from european_values.data_processing import process_data
from european_values.utils import apply_subset_filtering

logger = logging.getLogger("plot_typicality_diagnostics")


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float).reshape(-1)
    values = np.sort(values)
    probs = np.arange(1, values.size + 1) / values.size
    return values, probs


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
    df, _ = process_data(df=df, config=config, normalize=False)

    question_cols = [col for col in df.columns if col.startswith("question_")]
    responses = df[question_cols].values

    logger.info("Loading pipeline...")
    pipeline = joblib.load(config.typicality.model_path)
    scores = pipeline.transform(responses)
    df = df.copy()
    df["score"] = scores

    output_dir = Path(config.typicality.diagnostics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bins = int(config.typicality.diagnostics_bins)
    group_col = "country_group" if config.use_country_groups else "country_code"

    eu_scores = df.query("country_group == 'EU'")["score"].values
    non_eu_scores = df.query("country_group != 'EU'")["score"].values

    logger.info("Plotting score distributions...")
    plt.figure(figsize=(8, 5))
    plt.hist(eu_scores, bins=bins, alpha=0.6, label="EU", density=True)
    plt.hist(non_eu_scores, bins=bins, alpha=0.6, label="Non-EU", density=True)
    plt.title("Score distribution (EU vs Non-EU)")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_hist_eu_vs_noneu.png", dpi=150)
    plt.close()

    logger.info("Plotting score ECDFs...")
    plt.figure(figsize=(8, 5))
    for label, values in [("All", df["score"].values), ("EU", eu_scores), ("Non-EU", non_eu_scores)]:
        x, y = _ecdf(values)
        plt.plot(x, y, label=label)
    plt.title("Score ECDF")
    plt.xlabel("Score")
    plt.ylabel("ECDF")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_ecdf.png", dpi=150)
    plt.close()

    logger.info("Plotting group boxplots...")
    group_counts = df[group_col].value_counts()
    top_groups = group_counts.head(int(config.typicality.diagnostics_max_boxplot_groups))
    boxplot_groups = top_groups.index.tolist()
    boxplot_values = [df.query(f"{group_col} == @grp")["score"].values for grp in boxplot_groups]
    plt.figure(figsize=(max(10, len(boxplot_groups) * 0.35), 6))
    plt.boxplot(boxplot_values, labels=boxplot_groups, showfliers=False)
    plt.title(f"Score distribution by {group_col} (top {len(boxplot_groups)})")
    plt.xlabel(group_col)
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / f"score_boxplot_{group_col}.png", dpi=150)
    plt.close()

    raw_scores = None
    if hasattr(pipeline, "named_steps"):
        if "scaler" in pipeline.named_steps and "model" in pipeline.named_steps:
            logger.info("Computing raw model scores for diagnostics...")
            scaled = pipeline.named_steps["scaler"].transform(responses)
            raw_scores = pipeline.named_steps["model"].transform(scaled)

    if raw_scores is not None:
        raw_scores = np.asarray(raw_scores, dtype=float).reshape(-1)

        plt.figure(figsize=(8, 5))
        plt.hist(raw_scores, bins=bins, alpha=0.8, density=True)
        plt.title("Raw model score distribution")
        plt.xlabel("Raw score")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(output_dir / "raw_score_hist.png", dpi=150)
        plt.close()

        max_points = int(config.typicality.diagnostics_scatter_points)
        rng = np.random.default_rng(4242)
        if raw_scores.size > max_points:
            idx = rng.choice(raw_scores.size, size=max_points, replace=False)
            scatter_raw = raw_scores[idx]
            scatter_scores = scores[idx]
        else:
            scatter_raw = raw_scores
            scatter_scores = scores

        plt.figure(figsize=(7, 5))
        plt.scatter(scatter_raw, scatter_scores, s=6, alpha=0.4)
        plt.title("Raw score vs final score")
        plt.xlabel("Raw score")
        plt.ylabel("Final score")
        plt.tight_layout()
        plt.savefig(output_dir / "raw_vs_score_scatter.png", dpi=150)
        plt.close()

    logger.info("Writing summary statistics...")
    summary = (
        df.groupby(group_col)["score"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
            p10=lambda s: s.quantile(0.1),
            p90=lambda s: s.quantile(0.9),
        )
        .sort_values(by="mean", ascending=False)
    )
    summary.to_csv(output_dir / f"score_summary_{group_col}.csv")
    logger.info(f"Diagnostics saved to {output_dir.as_posix()}")


if __name__ == "__main__":
    main()
