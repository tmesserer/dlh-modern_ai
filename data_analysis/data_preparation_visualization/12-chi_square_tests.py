#!/usr/bin/env python3
"""
Module to perform Chi-Square tests of independence on categorical features.
"""
import pandas as pd
from scipy import stats


def chi_square_tests(df):
    """
    Performs chi-square tests for categorical features, using SciPy.
    This function calculates the Chi-square test of independence to determine
    if there is a significant relationship between each categorical feature
    and the target 'Churn' variable.
    Args:
        df (pandas.DataFrame) with Churn, categorical and numerical columns.
    Return:
        A dictionary, mapping feature names to their Chi-square p-values.
    """
    if (type(df).__name__ != "DataFrame" or df.empty
            or "Churn" not in df.columns):
        return 1
    categorical_cols = df.select_dtypes(include=['object']).columns
    features = [col for col in categorical_cols
                if col not in ['Churn', 'customerID']]
    results = {}
    for col in features:
        subset = df[[col, 'Churn']].dropna()
        contingency_table = pd.crosstab(subset[col], subset['Churn'])
        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
            _, p, _, _ = stats.chi2_contingency(contingency_table)
            results[col] = p
    return results