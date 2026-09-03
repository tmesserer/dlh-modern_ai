#!/usr/bin/env python3
"""Module for building decision tree models using scikit-learn"""
from sklearn import tree
from sklearn import export_text


def draw(clf, feature_names, class_names):
    """displays the textual structure of a trained decision tree classifier using Scikit-learn.
    
    Args:
        clf (DecisionTreeClassifier): The decision tree classifier to be trained.
        feature_names: A list of the input feature names
        class_names: A list of the target class names

    Returns:
        None: The function trains the model in place and does not return a value.
    """
    clf = clf.export_text(clf, feature_names=feature_names, class_names=class_names)
    return None