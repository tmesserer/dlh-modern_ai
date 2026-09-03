#!/usr/bin/env python3
"""Module for building decision tree models using scikit-learn"""
from sklearn import model_selection


def get_pruning_path(clf, X, y):
    """ retrieves the cost-complexity pruning path for a given decision tree classifier.

    Arguments:

    clf: A DecisionTreeClassifier instance
    X: Input features
    y: Target labels
    Returns:

    ccp_alphas: A NumPy array containing the effective alpha values used for pruning
    impurities: A NumPy array containing the total impurity of leaves at each corresponding alpha
    """
    my_param_grid = {"criterion": "gini",
                  "max_depth": [2, 3, 4, 5]
                  "min_samples_leaf": [2, 3, 4, 5]
                  "min_samples_split": [2, 3, 4, 5]
                  }
    model_selection.GridSearchCV(clf, param_grid=my_param_grid)
    clf.fit(X, y)
    impurities = 
    ccp_alphas =

    return ccp_alphas, impurities