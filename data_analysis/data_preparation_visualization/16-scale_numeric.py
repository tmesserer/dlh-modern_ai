#!/usr/bin/env python3
"""
Module to scale numeric features for modeling.
"""
from sklearn import preprocessing


def scale_numeric(df):
    """
    Standardize MonthlyCharges and TotalCharges using StandardScaler.
    Args:
        df (pandas.DataFrame): The input DataFrame.
    Returns:
        pandas.DataFrame: The DataFrame with MonthlyCharges and
            TotalCharges scaled to mean=0, std=1.
    """
    if type(df).__name__ != "DataFrame" or df.empty:
        return 1
    numeric_cols = ['MonthlyCharges', 'TotalCharges']
    if not all(col in df.columns for col in numeric_cols):
        return 1
    df_copy = df.copy()
    scaler = preprocessing.StandardScaler()
    df_copy[numeric_cols] = scaler.fit_transform(df_copy[numeric_cols])
    return df_copy