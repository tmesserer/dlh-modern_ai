#!/usr/bin/env python3
"""
Module to create engineered features for the churn dataset.
"""
import pandas as pd


def create_features(df):
    """
    Create NumServices and TenureGroup features.
    NumServices counts subscribed services across selected service-related
    columns. For InternetService, DSL and Fiber optic are treated as subscribed
    services, while No is not.
    TenureGroup bins tenure into 0-12, 13-24, 25-48, 49-60, and 60+.
    Args:
        df (pandas DataFrame): The input DataFrame.
    Returns:
        The modified DataFrame.
    """
    if (not isinstance(df, pd.DataFrame) or df.empty
            or "tenure" not in df.columns):
        return 1
    service_cols = [
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    available_service_cols = [col for col in service_cols if col in df.columns]
    df["NumServices"] = 0
    for col in available_service_cols:
        if col == "InternetService":
            df["NumServices"] += (
                df[col].isin(["DSL", "Fiber optic"]).astype(int)
            )
        else:
            df["NumServices"] += df[col].eq("Yes").astype(int)
    tenure_bins = [0, 12, 24, 48, 60, float("inf")]
    tenure_labels = ["0-12", "13-24", "25-48", "49-60", "60+"]
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=tenure_bins,
        labels=tenure_labels,
        include_lowest=True,
        right=True
    )
    cols_to_drop = [col for col in available_service_cols if col in df.columns]
    if "tenure" in df.columns:
        cols_to_drop.append("tenure")
    df.drop(columns=cols_to_drop, inplace=True)
    return df