# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # ***************************************************
    # Use normal equations: w = (X^T X)^(-1) X^T y
    # For numerical stability, use np.linalg.solve instead of np.linalg.inv
    
    # Compute X^T X
    tx = np.array(tx)
    y = np.array(y)

    gram_matrix = tx.T.dot(tx)
    
    # Compute X^T y
    xy = tx.T.dot(y)
    
    # Solve the system (X^T X) w = X^T y
    w = np.linalg.solve(gram_matrix, xy)
    
    # Calculate MSE
    mse = 0
    for i in range (y.shape[0]):
        mse += (y-tx.dot(w))**2
    
    return w, mse
    # returns w (optimal weights), mse (mean squared error)
    # ***************************************************