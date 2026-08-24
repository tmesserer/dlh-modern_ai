#!/usr/bin/env python3
"""Module to encode categorical features for machine learning models."""

import pandas as pd
from sklearn import preprocessing


def encode_features(df):
    """
    Encodes features for modeling using Scikit-learn.
    Args:
        df (pandas.DataFrame): The input DataFrame.
    Returns:
        tuple: (encoded DataFrame, fitted LabelEncoder,
        fitted binary OrdinalEncoder, fitted TenureGroup OrdinalEncoder)
    """
    if not isinstance(df, pd.DataFrame) or df.empty or len(df.columns) == 0:
        return 1
    binary_cols = [
        "Partner",
        "Dependents",
        "PaperlessBilling",
        "SeniorCitizen"
    ]
    required = ["Churn", "Contract", "PaymentMethod", "TenureGroup"]
    required += binary_cols
    if not all(col in df.columns for col in required):
        return 1
    df = df.copy()
    # Step 1: LabelEncoder for Churn
    churn_le = preprocessing.LabelEncoder()
    df["Churn"] = churn_le.fit_transform(df["Churn"])
    # Step 2: OrdinalEncoder for binary columns
    binary_oe = preprocessing.OrdinalEncoder(categories=[['No', 'Yes']])
    for col in binary_cols:
        df[col] = binary_oe.fit_transform(df[[col]]).astype(int)
    # Step 3: One-hot encoding for Contract and PaymentMethod
    df = pd.get_dummies(
        df,
        columns=['Contract', 'PaymentMethod'],
        drop_first=True,
        dtype=int
    )
    # Step 4: OrdinalEncoder for TenureGroup
    tenure_oe = preprocessing.OrdinalEncoder()
    df["TenureGroup"] = tenure_oe.fit_transform(
        df[["TenureGroup"]]
    ).astype("int")
    return df, churn_le, binary_oe, tenure_oe