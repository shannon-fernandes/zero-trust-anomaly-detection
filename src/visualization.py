"""
visualization.py

Graphs are generated only from results passed in by evaluation.py
and zero_trust.py. No result values are manually fabricated.

If data_source != "real", every figure is stamped "SYNTHETIC DEMO DATA"
to prevent it from being mistaken for a real experimental result.
"""

from pathlib import Path

import matplotlib

# Use a non-interactive backend to save figures directly to files.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from synthetic_data import SYNTHETIC_NOTICE


def _finish(fig, title, data_source, path):
    """
    Add the title and synthetic-data watermark, then save the figure.

    The output directory is created automatically. The output path is
    converted to an absolute path before saving to avoid path ambiguity.
    """

    suffix = (
        ""
        if data_source == "real"
        else f"\n[{SYNTHETIC_NOTICE}]"
    )

    fig.suptitle(
        title + suffix,
        fontsize=11,
        color="black" if data_source == "real" else "red"
    )

    if data_source != "real":
        fig.text(
            0.5,
            0.5,
            "SYNTHETIC DEMO",
            fontsize=30,
            color="red",
            alpha=0.12,
            ha="center",
            va="center",
            rotation=30
        )

    fig.tight_layout()

    # Resolve the destination and create its parent directory.
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Explicitly save as PNG.
        fig.savefig(
            str(output_path),
            format="png",
            dpi=150,
            bbox_inches="tight"
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Could not save graph to: {output_path}\n"
            f"Check whether the file is open or locked by another program, "
            f"and whether the output folder is writable.\n"
            f"Original error: {exc}"
        ) from exc
    finally:
        plt.close(fig)

    return output_path


def plot_confusion_matrix(
    metrics,
    path,
    title,
    data_source="synthetic"
):
    """Plot the confusion matrix."""

    cm = np.array(metrics["confusion_matrix"])

    fig, ax = plt.subplots(figsize=(4.5, 4.2))

    ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1], ["Normal", "Attack"])
    ax.set_yticks([0, 1], ["Normal", "Attack"])

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if cm[i, j] > cm.max() / 2
                    else "black"
                )
            )

    return _finish(fig, title, data_source, path)


def plot_metrics_comparison(
    named_metrics,
    path,
    title,
    data_source="synthetic"
):
    """Compare accuracy, precision, recall, and F1 scores."""

    names = list(named_metrics)

    keys = [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]

    labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]

    x = np.arange(len(keys))
    width = 0.8 / max(len(names), 1)

    fig, ax = plt.subplots(figsize=(7, 4.2))

    for i, name in enumerate(names):
        vals = [
            named_metrics[name][key]
            for key in keys
        ]

        bars = ax.bar(
            x + i * width,
            vals,
            width,
            label=name
        )

        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.2f}",
                ha="center",
                fontsize=7
            )

    ax.set_xticks(
        x + width * (len(names) - 1) / 2,
        labels
    )

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.legend(fontsize=8)

    return _finish(fig, title, data_source, path)


def plot_decision_distribution(
    summary,
    path,
    title,
    data_source="synthetic"
):
    """Plot the distribution of ALLOW, FLAG, and DENY decisions."""

    names = ["ALLOW", "FLAG", "DENY"]
    decisions = summary["decisions"]

    has_split = (
        "attacks" in decisions["ALLOW"]
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    if has_split:
        normal = [
            decisions[name]["normal"]
            for name in names
        ]

        attacks = [
            decisions[name]["attacks"]
            for name in names
        ]

        ax.bar(
            names,
            normal,
            label="Actually normal",
            color="tab:green"
        )

        ax.bar(
            names,
            attacks,
            bottom=normal,
            label="Actually attack",
            color="tab:red"
        )

        ax.legend(fontsize=8)

    else:
        counts = [
            decisions[name]["count"]
            for name in names
        ]

        ax.bar(names, counts)

    ax.set_ylabel("Number of flows")

    return _finish(fig, title, data_source, path)


def plot_risk_histogram(
    scores,
    y_true,
    path,
    title,
    data_source="synthetic",
    thresholds=None
):
    """Plot risk-score distributions for normal and attack traffic."""

    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y_true)

    if len(scores) != len(y):
        raise ValueError(
            "scores and y_true must have the same length."
        )

    fig, ax = plt.subplots(figsize=(6, 4))

    bins = np.linspace(0, 1, 21)

    ax.hist(
        scores[y == 0],
        bins=bins,
        alpha=0.6,
        label="Normal",
        color="tab:green"
    )

    ax.hist(
        scores[y == 1],
        bins=bins,
        alpha=0.6,
        label="Attack",
        color="tab:red"
    )

    if thresholds is not None:
        ax.axvline(
            thresholds["allow_below"],
            linestyle="--",
            color="orange",
            label="ALLOW/FLAG"
        )

        ax.axvline(
            thresholds["deny_at_or_above"],
            linestyle="--",
            color="black",
            label="FLAG/DENY"
        )

    ax.set_xlabel("Risk score")
    ax.set_ylabel("Number of flows")
    ax.legend(fontsize=8)

    return _finish(fig, title, data_source, path)


def plot_threshold_sweep(
    sweep_df,
    path,
    title,
    metric="f1",
    data_source="synthetic"
):
    """
    Plot evaluation scores across different ALLOW and DENY thresholds.
    """

    required_columns = {
        "allow_below",
        "deny_at_or_above",
        metric
    }

    missing_columns = required_columns - set(
        sweep_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Threshold sweep is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    pivot = sweep_df.pivot(
        index="allow_below",
        columns="deny_at_or_above",
        values=metric
    )

    # Sort the thresholds so the graph axes are consistent.
    pivot = pivot.sort_index().sort_index(axis=1)

    values = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(
        values,
        cmap="viridis",
        aspect="auto",
        origin="lower"
    )

    ax.set_xticks(
        np.arange(len(pivot.columns)),
        [f"{value:g}" for value in pivot.columns]
    )

    ax.set_yticks(
        np.arange(len(pivot.index)),
        [f"{value:g}" for value in pivot.index]
    )

    ax.set_xlabel("DENY threshold")
    ax.set_ylabel("ALLOW threshold")

    ax.set_title(f"Threshold Sweep: {metric.upper()}")

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]

            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=7
                )

    fig.colorbar(
        im,
        ax=ax,
        label=metric.upper()
    )

    return _finish(fig, title, data_source, path)


def plot_category_detection(
    per_cat,
    path,
    title,
    data_source="synthetic"
):
    """Plot the fraction of each traffic category flagged as an attack."""

    if not per_cat:
        raise ValueError(
            "per_cat is empty; no category data is available to plot."
        )

    cats = sorted(
        per_cat,
        key=lambda category: (
            per_cat[category]["flagged_as_attack_rate"]
        )
    )

    rates = [
        per_cat[category]["flagged_as_attack_rate"]
        for category in cats
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.barh(
        cats,
        rates,
        color="tab:purple"
    )

    ax.set_xlim(0, 1.05)

    ax.set_xlabel(
        "Fraction flagged as attack "
        "(Normal row = false-alarm rate)"
    )

    ax.set_ylabel("Traffic category")

    return _finish(fig, title, data_source, path)