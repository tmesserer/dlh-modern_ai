#!/usr/bin/env python3
"""Module for building decision tree models using scikit-learn"""
from sklearn import tree


def build_decision_tree(min_samples_leaf, min_samples_split, random_state):
    """Build a decision tree model with the specified parameters.
    
    Args:
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        random_state (int): The seed used by the random number generator.

    Returns:
        DecisionTreeClassifier: The constructed decision tree model.
    """
    clf = tree.DecisionTreeClassifier(min_samples_leaf=min_samples_leaf,
                                    min_samples_split=min_samples_split,
                                    random_state=random_state)
    return clf
    