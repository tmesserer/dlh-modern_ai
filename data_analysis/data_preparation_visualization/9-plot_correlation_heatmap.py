#!/usr/bin/env python3
"""Module to visualize correlations between continuous numeric features."""

import seaborn as sns
import matplotlib.pyplot as plt


def plot_correlation_heatmap(df):
    """
    Computes pairwise correlations for continuous numeric features
    and generates an annotated heatmap.
    Args:
        df (pandas.DataFrame): The input dataframe containing the data.
    Returns:
        None
        Displays the correlation heatmap.
        The darker the hue, the highest the correlation.
    """
    if type(df).__name__ != "DataFrame":
        return 1
    plt.figure(figsize=(6, 5))
    data = df.select_dtypes(include=["number"])
    if data.empty or len(data.columns) < 2:
        return 1
    correlation_matrix = data.corr()
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1
    )
    plt.title("Correlation Matrix")
    plt.show()
    return 0
