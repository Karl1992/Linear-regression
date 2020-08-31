"""
Do not change the input and output format.
If our script cannot run your code or the format is improper, your code will not be graded.

The only functions you need to implement in this template is linear_regression_noreg, linear_regression_invertible，regularized_linear_regression,
tune_lambda, test_error and mapping_data.
"""

import numpy as np
import pandas as pd

###### Q1.1 ######
def mean_absolute_error(w, X, y):
    """
    Compute the mean absolute error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean absolute error
    """
    #####################################################
    # TODO 1: Fill in your code here #
    #####################################################
    n = y.shape[0]
    y_hat = []
    inter_matrix = w * X
    for i in range(len(inter_matrix)):
        y_hat.append(sum(inter_matrix[i]))
    y_hat = np.array(y_hat)
    err = sum(abs(y_hat - y))/n
    return err

###### Q1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here #
  #####################################################
  X_T = X.transpose()
  w = np.dot(np.dot(np.linalg.inv((np.dot(X_T,X))), X_T),y)
  return w

###### Q1.3 ######
def linear_regression_invertible(X, y):
    """
    Compute the weight parameter given X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    Returns:
    - w: a numpy array of shape (D, )
    """
    #####################################################
    # TODO 3: Fill in your code here #
    #####################################################
    D = X.shape[1]
    I = np.eye(D, dtype=int)
    M = np.dot(X.transpose(), X)
    eigen_value, eigen_vector = np.linalg.eig(M)
    threshold = min(abs(np.array(eigen_value)))
    while threshold < 10**(-5):
        M = M + 10**(-1) * I
        eigen_value, eigen_vector = np.linalg.eig(M)
        threshold = min(abs(np.array(eigen_value)))
    w = np.dot(np.dot(np.linalg.inv(M), X.transpose()), y)
    return w

###### Q1.4 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here #
  #####################################################		
    D = X.shape[1]
    I = np.eye(D, dtype=int)
    M = np.dot(X.transpose(), X)
    w = np.dot(np.dot(np.linalg.inv(M + lambd * I), X.transpose()), y)
    return w

###### Q1.5 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
    #####################################################
    # TODO 5: Fill in your code here #
    #####################################################
    index = np.arange(-19, 20, 1, dtype=float)
    bestlambda = -99
    least_mae = None
    for i in index:
        lambd = 10**i
        w = regularized_linear_regression(Xtrain, ytrain, lambd)
        mae = mean_absolute_error(w, Xval, yval)
        if least_mae == None:
            least_mae = mae
        else:
            if mae < least_mae:
                least_mae = mae
                bestlambda = lambd
    return bestlambda
    

###### Q1.6 ######
def mapping_data(X, power):
    """
    Mapping the data.
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training feature.
    - power: A integer that indicate the power in polynomial regression
    Returns:
    - X: mapped_X, You can manully calculate the size of X based on the power and original size of X
    """
    #####################################################
    # TODO 6: Fill in your code here #
    #####################################################		
    power_index = np.arange(1, power+1, 1, dtype = float)
    mapping_X = X
    for power in power_index:
        if power ==1:
            continue
        else:
            X_power = np.power(X, power, dtype = float)
            mapping_X = np.append(mapping_X,X_power,axis = 1)
    return mapping_X


