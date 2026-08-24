#!/usr/bin/env python3
"""
Module to remove non-predictive identifier columns from a DataFrame.
"""


def drop_customerID(df):
    """
    Drops the customerID column from a pandas DataFrame.
    Args:
        df (pandas.DataFrame): The DataFrame to process.
    Returns:
        pandas.DataFrame: The modified DataFrame without the customerID column.
    """
    if type(df).__name__ != "DataFrame" or "customerID" not in df.columns:
        return 1
    return df.drop(columns=["customerID"])
