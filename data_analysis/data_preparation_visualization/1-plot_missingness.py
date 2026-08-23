#!/usr/bin/env python3
"""Module for data preparation and visualization"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_missingness(df):
    """function that visualizes missing values in a DataFrame"""
    plt.figure(figsize=(12, 8))

    """ my try:
    # plt.plot(df.loc[:].isna(), df.columns, marker='|')
    # plt.yticks(df.columns)
    for col in df.columns:
        mask = df[col].isna()
        missing_rows = df.index[mask]
        plt.scatter(missing_rows, [col] * len(missing_rows),
                    marker='|', color='C0')
    plt.yticks(df.columns)
    plt.tight_layout()
    plt.show()
    """
    df_nul = df.isnull()
    # print(df_nul)  # helper
    nul = np.where(df_nul)
    # print(nul)  # 'helper'

    x = nul[0]
    y = nul[1]

    plt.scatter(x, y, marker='|')
    # print(df.columns.values)

    plt.title("Missingness Plot")
    plt.yticks(np.arange(0, len(df.columns.values)), df.columns.values)

    plt.tight_layout()
    plt.show()
    return None
