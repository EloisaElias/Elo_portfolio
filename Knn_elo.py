from __future__ import division
from collections import Counter
from itertools import izip
import numpy as np


class Knn(object):
  """docstring for Knn
  kNN:
    for every point in the dataset:
    calculate the distance between the point and x
    sort the distances in increasing order
    take the k items with the smallest distances to x
    return the majority class among these items
  """

  def __init__(self, k=3, distance_f=None):

    self.k = k
    self.distance = distance_f
    self.X = None
    self.y = None

  def fit(self, X, y):
    self.X = X
    self.y = y

  def predict(self, X):
    if (self.X is None) or (self.y is None):
      raise ValueError('Please run .fit(X, y) method first')

    predictions = []
    for vectpoint in X:
      distances = []
      for row, label in izip(self.X, self.y):
        distances.append((label, self.distance(vectpoint, row)))
      sorted_distances = sorted(distances, key= lambda x: x[1])[:self.k]
      major = Counter(closest[0] for closest in sorted_distances).most_common(1)[0][0]
      predictions.append(major)

    return np.array(predictions)

  def score(self, X, y):
    if len(X) != len(y):
      raise ValueError('Shape X: {} is not compatible with shape  y: {}'.format(X.shape, y.shape))

    predictions = self.predict(X)
    accuracy = sum(y == predictions) / float(len(y))

    return accuracy

