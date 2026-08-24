#!/usr/bin/env python3
"""
Module to handle removing duplicate rows from a DataFrame.
"""


def remove_duplicates(df):
    """
    Drops all duplicate rows from a pandas DataFrame.
    Args:
        df (pandas.DataFrame): The DataFrame to process.
    Returns:
        pandas.DataFrame: The deduplicated DataFrame.
    """
    if type(df).__name__ != "DataFrame":
        return 1
    return df.drop_duplicates()