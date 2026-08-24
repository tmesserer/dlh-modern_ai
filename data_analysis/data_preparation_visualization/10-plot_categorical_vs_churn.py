#!/usr/bin/env python3
"""Module to visualize categorical features against churn rates."""
import matplotlib.pyplot as plt


def plot_categorical_vs_churn(df, col):
    """
    Visualizes churn rates per category.
    Args:
        df (pandas.DataFrame): The input DataFrame containing the dataset.
        col (str): The target categorical column name.
    Returns:
        None
        The function displays the generated bar plot directly.
    """
    if (type(df).__name__ != "DataFrame" or df.empty
            or "Churn" not in df.columns or col not in df.columns):
        return 1
    plt.figure(figsize=(12, 8))
    churn_rate = (df["Churn"] == "Yes").groupby(df[col]).mean()
    plt.bar(churn_rate.index, churn_rate.values)
    plt.title(f"Churn Rate by {col}")
    plt.ylabel("Churn Rate")
    plt.xticks(rotation=45)
    plt.show()
    return 0
