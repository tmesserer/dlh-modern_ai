#!/usr/bin/env python3
"""Module for building decision tree models using scikit-learn"""
from sklearn import tree


def train_tree(clf, X, y):
    """Train a decision tree model with the specified parameters.
    
    Args:
        clf (DecisionTreeClassifier): The decision tree classifier to be trained.
        X (array-like): The feature matrix for training.
        y (array-like): The target vector for training.

    Returns:
        None: The function trains the model in place and does not return a value.
    """
    clf = clf.fit(X, y)
    return None