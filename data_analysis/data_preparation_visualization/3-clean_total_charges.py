#!/usr/bin/env python3
"""Module for data preparation and visualization"""
import pandas as pd


def clean_total_charges(df, method='drop'):
    """function that handles missing values in TotalCharges"""
    if method == 'drop':
        df["TotalCharges"] = df["TotalCharges"].isna().drop() # needs fixing

    elif method == 'median':
        tc_median = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(tc_median)

    elif method == 'impute':
        df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

    else:
        return df
        
    return df


# df = pd.read_csv('Telco-Customer-Churn.csv')
# print(df.head())
