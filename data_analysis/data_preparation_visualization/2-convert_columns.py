#!/usr/bin/env python3
"""Module for data preparation and visualization"""
import pandas as pd


def convert_columns(df):
    """function that performs type conversion for specific columns"""
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes"})
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

    return df
