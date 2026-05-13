"""Sweep-specific plots: how a metric varies across parameter values."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


METRICS = [
    ("active_count", "Active players (final day)"),
    ("true_skill_mean", "True skill mean (final day)"),
    ("rating_error_mean", "Rating error (final day)"),
    ("gear_mean", "Gear mean (final day)"),
    ("blowout_share", "Blowout share (final day)"),
]


def _final_row(path: Path) -> dict:
    df = pl.read_parquet(path)
    last_day = df.select(pl.col("day").max()).item()
    row = df.filter(pl.col("day") == last_day).to_dicts()[0]
    matches = float(row.get("matches_played") or 0.0)
    blowouts = float(row.get("blowouts") or 0.0)
    row["blowout_share"] = (blowouts / matches) if matches > 0 else float("nan")
    return row


def _load_sweep(sweep_dir: Path) -> tuple[dict, list[tuple[list[float], dict]]]:
    metadata = json.loads((sweep_dir / "sweep.json").read_text())
    points_dir = sweep_dir / "points"
    data: list[tuple[list[float], dict]] = []
    for point in metadata["points"]:
        experiment_name = point["experiment_name"]
        version = point["experiment_version"]
        agg_path = points_dir / experiment_name / version / "aggregate.parquet"
        row = _final_row(agg_path)
        values = [point["overrides"][p] for p in metadata["parameters"]]
        data.append((values, row))
    return metadata, data


def plot_sweep(sweep_dir: Path) -> None:
    sweep_dir = Path(sweep_dir)
    plots_dir = sweep_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    metadata, data = _load_sweep(sweep_dir)
    if len(metadata["parameters"]) == 2 and metadata["mode"] == "grid":
        _plot_2d_heatmap(metadata, data, plots_dir)
    else:
        _plot_1d(metadata, data, plots_dir)


def _plot_1d(metadata, data, plots_dir):
    xs = np.array([d[0][0] for d in data], dtype=float)
    param_label = metadata["parameters"][0].rsplit(".", 1)[-1]
    fig, axes = plt.subplots(
        len(METRICS), 1, figsize=(8, 2.5 * len(METRICS)), sharex=True
    )
    for ax, (key, title) in zip(axes, METRICS):
        ys = np.array(
            [d[1].get(key, float("nan")) for d in data], dtype=float
        )
        order = np.argsort(xs)
        ax.plot(xs[order], ys[order], marker="o", linewidth=2)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel(param_label)
    fig.suptitle(f"sweep: {metadata['name']}")
    fig.tight_layout()
    fig.savefig(plots_dir / "final_metrics.png", dpi=120)
    plt.close(fig)


def _plot_2d_heatmap(metadata, data, plots_dir):
    ax_params = metadata["parameters"]
    x_values, y_values = (sorted(set(v)) for v in metadata["value_lists"])
    x_idx = {v: i for i, v in enumerate(x_values)}
    y_idx = {v: i for i, v in enumerate(y_values)}
    fig, axes = plt.subplots(
        1, len(METRICS), figsize=(4 * len(METRICS), 4), squeeze=False
    )
    for ax, (key, title) in zip(axes[0], METRICS):
        grid = np.full((len(y_values), len(x_values)), float("nan"))
        for values, row in data:
            xi = x_idx[values[0]]
            yi = y_idx[values[1]]
            grid[yi, xi] = row.get(key, float("nan"))
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels([f"{v:g}" for v in x_values], rotation=45)
        ax.set_yticks(range(len(y_values)))
        ax.set_yticklabels([f"{v:g}" for v in y_values])
        ax.set_xlabel(ax_params[0].rsplit(".", 1)[-1])
        ax.set_ylabel(ax_params[1].rsplit(".", 1)[-1])
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"sweep: {metadata['name']}")
    fig.tight_layout()
    fig.savefig(plots_dir / "heatmaps.png", dpi=120)
    plt.close(fig)
