#!/usr/bin/env python3
"""Module for visualizing continuous distributions of numerical features."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_continuous_distributions(df, columns_to_plot=None):
    """
    Visualizes the continuous distributions of selected numerical columns.
    For each specified column, this function generates a side-by-side plot:
    1. A histogram with an overlaid Kernel Density Estimation (KDE) curve.
    2. A horizontal boxplot showing the median, quartiles, and outliers.
    Args:
        - df (pandas.DataFrame): the DataFrame containing the numeric data.
        - columns_to_plot (list of str, optional): A list of continuous numeric
        columns to visualize. If None, the function automatically identifies
        and plots all numeric columns in the DataFrame.
    Returns
    -   None
        The function displays the plot grid and saves it to 'Task_8.png'.
    """
    if type(df).__name__ != "DataFrame" or df.empty:
        return 1
    if columns_to_plot is None:
        cols = df.select_dtypes(include=["number"]).columns.tolist()
        columns_to_plot = cols
    else:
        columns_to_plot = [col for col in columns_to_plot if col in df.columns]
    if not columns_to_plot:
        return 1
    n_cols = len(columns_to_plot)
    fig, axes = plt.subplots(n_cols, 2, figsize=(10, 3*n_cols))
    if n_cols == 1:
        axes = axes.reshape(1, -1)
    for i, col in enumerate(columns_to_plot):
        data = df[col].dropna()
        kde = stats.gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        axes[i, 0].hist(data, bins=30, density=True, alpha=0.7,
                        edgecolor="black")
        axes[i, 0].plot(x_range, kde(x_range), color="red", linestyle="--")
        axes[i, 0].set_title(f"{col} Histogram + KDE")
        axes[i, 1].boxplot(data, vert=False)
        axes[i, 1].set_title(f"{col} Boxplot")
    plt.tight_layout()
    plt.savefig("Task_8.png")
    plt.show()
    return 0
