#!/usr/bin/env python3
"""
Module to plot continuous numeric features segmented by churn status.
"""


import matplotlib.pyplot as plt


def plot_numeric_vs_churn(df, col):
    """
    Compares continuous numeric feature distributions by churn.
    The function splits the dataset into two subsets, w/ or w/o Churn.
    Args:
        df: pandas DataFrame with Churn column
        col: Numeric column name
    Return:
        None
        Plot the distribution by churn of the column.
    """
    if (type(df).__name__ != "DataFrame" or df.empty
            or "Churn" not in df.columns or col not in df.columns):
        return 1
    plt.figure(figsize=(12, 8))
    churn_no = df.loc[df["Churn"] == "No", col].dropna()
    churn_yes = df.loc[df["Churn"] == "Yes", col].dropna()
    plt.hist([churn_no, churn_yes], bins=30, label=["No", "Yes"])
    plt.title(f"{col} Distribution by Churn")
    plt.xlabel(col)
    plt.legend(title="Churn")
    plt.show()
    return 0