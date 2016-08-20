from __future__ import division

import numpy as np
import pandas as pd


def hypothesis(coeffs, X):
  '''Logistic function / Sigmoid function'''
  return 1./(1. + np.exp(-1 * np.dot(coeffs, np.transpose(X))))x


def predict(coeffs, X):
  '''Calculate the predicted values (0 or 1) for the given
  data with the given coefficients.
  Assuming threshold = 0.5'''
  return hypothesis(coeffs, X).round()

def log_likelihood(coeffs, X, y, lambd=0):
  '''Cost function, maximize the log likelihood + l2.
  L2 = LassoRidge'''

  l2_reg = lambd * sum(coeffs.T**2)
  hx = hypothesis(coeffs, X)
  cost = sum(y * np.log(hx) + (1 - y) * np.log(1 - hx))
  return cost + l2_reg

def log_likelihood_gradient(coeffs, X, y, lambd=0):
  '''Gradient function - gradient of the cost funtion'''
  l2_reg_der = 2 * lambd * sum(coeffs.T)
  h = hypothesis(coeffs, X)
  gradient = np.array([sum((y - h) * X[:, j]) for j in xrange(X.shape[1])])
  return gradient + l2_reg_der






















































