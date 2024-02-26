"""Utilities used for plotting purposes.

Author: Titouan Lorieul <titouan.lorieul@gmail.com>
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def escape_tex(s: str) -> str:
    """Escape special characters for LaTeX rendering."""
    if not plt.rcParams["text.usetex"]:
        return s

    s = s.replace("_", "\\_")
    s = s.replace("%", "\\%")
    s = s.replace("#", "\\#")
    return s


def plot_metric(df_metrics: pd.DataFrame, metric: str, ax: plt.Axis) -> plt.Axis:
    """Plot specific metric monitored during model training history.

    Parameters
    ----------
    df_metrics: pd.DataFrame containing metrics monitored during training
    metrics: name of the metric to plot
    ax: plt.Axis to use for plotting

    Returns
    -------
    ax: plt.Axis used for plotting
    """
    index = df_metrics.index
    index_name = index.name

    train_metric = df_metrics["train_" + metric]
    x = index[~train_metric.isnull()].drop_duplicates()
    train_metric = train_metric[~train_metric.isnull()]
    train_metric = (
        train_metric.reset_index()
        .drop_duplicates(index_name, keep="last")
        .set_index(index_name)
    )
    ax.plot(x, train_metric, color="b", label="train")

    if "val_" + metric in df_metrics:
        val_metric = df_metrics["val_" + metric]
        x = index[~val_metric.isnull()].drop_duplicates()
        val_metric = val_metric[~val_metric.isnull()]
        val_metric = (
            val_metric.reset_index()
            .drop_duplicates(index_name, keep="last")
            .set_index(index_name)
        )
        ax.plot(x, val_metric, color="g", label="validation")

    if index.name:
        ax.set_xlabel(escape_tex("# {}").format(index.name))

    ax.set_title(escape_tex(metric))

    ax.autoscale(axis="x", tight=True)
    ax.grid()
    ax.legend()

    return ax


def plot_history(
    df_metrics: pd.DataFrame,
    *,
    fig: Optional[plt.Figure] = None,
    axes: Optional[list[plt.Axis]] = None,
) -> tuple[plt.Figure, list[plt.Axis]]:
    """Plot model training history.

    Parameters
    ----------
    df_metrics: pd.DataFrame containing metrics monitored during training
    fig: plt.Figure to use for plotting
    axes: list of plt.Axis to use for plotting

    Returns
    -------
    fig: plt.Figure used for plotting
    axes: list of plt.Axis used for plotting
    """
    base_metrics = list(filter(lambda x: "val_" in x, df_metrics.keys()))
    base_metrics = [column[4:] for column in df_metrics.keys() if "val_" in column]

    if axes is None:
        ncols = 2
        nrows = int(np.ceil(len(base_metrics) / float(ncols)))
        figsize = (7, 2.5 * nrows)

        if fig is None:
            fig = plt.figure(figsize=figsize, constrained_layout=True)

        axes = fig.subplots(nrows=nrows, ncols=ncols)

        empty_axes = axes.ravel()[-(len(axes) - len(base_metrics) + 1):]
        for ax in empty_axes:
            ax.axis("off")

    for metric, ax in zip(base_metrics, axes.ravel()):
        ax.cla()
        ax = plot_metric(df_metrics, metric, ax)

    return fig, axes


if __name__ == "__main__":
    import argparse

    import pandas as pd

    parser = argparse.ArgumentParser(description="plots the training curves")
    parser.add_argument(
        "metrics_csv",
        nargs="+",
        help="training CSV file",
    )
    parser.add_argument(
        "--title",
        nargs="+",
        default=[""],
        help="plot title",
    )
    parser.add_argument(
        "--index_column",
        default="epoch",
        help="specify column to use as index",
    )
    args = parser.parse_args()

    for i, filename in enumerate(args.metrics_csv):
        title = args.title[i] if i < len(args.title) else ""
        df = pd.read_csv(filename, index_col=args.index_column)

        fig_hist, _ = plot_history(df)
        fig_hist.canvas.manager.set_window_title(filename)

        if title:
            fig_hist.suptitle(title)

    plt.show()
