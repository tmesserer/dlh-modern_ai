#!/usr/bin/env python3
"""
Module to clean missing values in the TotalCharges column.
"""


def clean_total_charges(df, method="drop"):
    """
    Handles missing values in the TotalCharges column using various strategies.
    Args:
        df (pandas.DataFrame): The DataFrame to modify.
        method (str): The strategy to use ('drop', 'median', 'impute').
    Returns:
        pandas.DataFrame: The modified DataFrame.
    """
    if type(df).__name__ != "DataFrame" or df.empty or len(df.columns) == 0:
        return 1
    if method not in ["drop", "median", "impute"]:
        return 1
    if "TotalCharges" not in df.columns:
        return 1
    if (method == "impute"
            and ("MonthlyCharges" not in df.columns
                 or "tenure" not in df.columns)):
        return 1
    df_copy = df.copy()
    if method == "drop":
        return df_copy.dropna(subset=["TotalCharges"])
    if method == "median":
        median = df_copy["TotalCharges"].median()
        df_copy.loc[:, "TotalCharges"] = df_copy["TotalCharges"].fillna(median)
    if method == "impute":
        imput = df_copy["MonthlyCharges"] * df_copy["tenure"]
        df_copy.loc[:, "TotalCharges"] = df_copy["TotalCharges"].fillna(imput)
    return df_copy
