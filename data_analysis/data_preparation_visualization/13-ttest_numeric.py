#!/usr/bin/env python3
"""
Module to perform Welch's t-tests on numeric features.
"""
from scipy import stats


def ttest_numeric(df):
    """
    Perform Welch's t-tests for continuous numeric features.
    The test compares the Churn=Yes and Churn=No groups for each numeric
    feature and returns the p-value for each feature.
    Args:
            df: pandas.DataFrame with a Churn column.
    Returns:
            A dictionary mapping feature names to Welch's t-test p-values.
    """
    if (type(df).__name__ != "DataFrame" or df.empty
            or "Churn" not in df.columns):
        return 1
    numeric_cols = df.select_dtypes(include=["number"]).columns
    results = {}
    for col in numeric_cols:
        churn_yes = df.loc[df["Churn"] == "Yes", col].dropna()
        churn_no = df.loc[df["Churn"] == "No", col].dropna()
        _, p_value = stats.ttest_ind(churn_yes, churn_no, equal_var=False)
        results[col] = p_value
    return results