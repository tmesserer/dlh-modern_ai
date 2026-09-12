#!/usr/bin/env python3
"""Module for k-means clustering"""
import numpy as np


def initialize(X, k):
    """function that initializes cluster centroids for K-means
    Args:
        X is a numpy.ndarray of shape (n, d) containing the dataset that
        will be used for K-means
         clustering:
            n is the number of data points
            d is the number of dimensions for each data point
        k is a positive integer containing the number of clusters

    Returns: a numpy.ndarray of shape (k, d) containing the initialized
    centroids for each cluster,
    or None on failure
    """
    try:
        min_X = np.min(X, axis=0)
        max_X = np.max(X, axis=0)
        size = [k, X.shape[1]]
        clusters = np.random.uniform(min_X, max_X, size)
        return clusters

    except (ValueError, TypeError, AttributeError):
        return None
