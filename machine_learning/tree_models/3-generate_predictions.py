#!/usr/bin/env python3
"""Module for building decision tree models using scikit-learn"""
from sklearn import tree
from sklearn import export_text


def generate_predictions(clf, X):
    """generate predictions from a trained tree-based classifier using Scikit-learn.
    
    Args:
        clf (DecisionTreeClassifier): A trained Scikit-learn classifier instance
        X: Feature matrix (NumPy array or pandas DataFrame)

    Returns:
        A NumPy array containing the predicted class labels for the input samples.
    """
    clf = clf.predict(X)
    return clf