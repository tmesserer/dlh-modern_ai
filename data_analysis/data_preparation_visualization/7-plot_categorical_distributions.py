#!/usr/bin/env python3
"""This module visualizes categorical feature distributions."""

import matplotlib.pyplot as plt


def plot_categorical_distributions(df, columns_to_plot=None):
    """
    Generates a grid of bar charts showing value counts for categorical
    features.
    Args:
        df (pandas.DataFrame): the input DataFrame.
        columns_to_plot (list of str, optional): specific columns to plot.
        If None, all columns with dtype object (excluding 'Churn') are used.
    Returns:
        None. Displays the plot grid and saves it as 'Task_7.png'.
    """
    if type(df).__name__ != "DataFrame" or df.empty:
        return 1
    if columns_to_plot is None:
        columns_to_plot = [
            col
            for col in df.columns
            if (str(df[col].dtype) in ("object", "str", "string")
                and col != "Churn")
        ]
    else:
        columns_to_plot = [c for c in columns_to_plot if c in df.columns]
    if not columns_to_plot:
        return 1
    n_cols, n_rows = 3, -(-len(columns_to_plot) // 3)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    for i, col in enumerate(columns_to_plot):
        counts = df[col].value_counts()
        axes[i].bar(counts.index.tolist(), counts.values)
        axes[i].set_title(col)
        axes[i].tick_params(axis="x", rotation=45)
    for j in range(len(columns_to_plot), len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.savefig("Task_7.png")
    plt.show()
    return 0
