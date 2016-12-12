import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

class Adaboostelo(object):
  def __init__(self, n_classifiers=50, learning_rate=1):
    self.base_classifier = DecisionTreeClassifier(max_depth=1)
    self.n_classifier = n_classifiers
    self.learning_rate = learning_rate
    # fit module:
    self.h_classifiers = []
    self.coefficient_weight = np.zeros(self.n_classifier, dtype=np.float)

  def fit(self, X, y):
    '''
    weight for each sample = w_i = sample_weight
    '''
    w_i = np.ones(len(X)) / float(len(X))
    # h(x) = classifier = Each weak learners produces
    # an output hypothesis h(x) for each sample in the training set
    for _classifier in xrange(self.n_classifier):
      cls, w_i, alpha = self._boost(X, y, w_i)
      self.h_classifiers.append(cls)
      self.coefficient_weight[_classifier] = alpha

  def _boost(self, X, y, sample_weight):
    '''
    sample_weight: At each iteration of the training process, a weight w_i
    is assigned to each sample in the training set equal to
    the current error E(F_{t-1}(x_{i})) on that sample.
    where: F_{t-1}(x_{i}) = Bost classifier = k_m(x_i)
    I = Incorrect predictions = I(yi !=k_m(xi))
    '''

    classifier  = clone(self.base_classifier)
    classifier.fit(X, y, sample_weight=sample_weight)
    predictions = classifier.predict(X)
    I = predictions != y
    error = np.sum(sample_weight[I]) / np.sum(sample_weight)
    alpha = self.learning_rate * np.log((1 - error) / error)
    sample_weight[I] *= np.exp(alpha)

    return classifier, sample_weight, alpha

  def predict(self, X):
    '''
    Suppose we have a data set {(x_{1},y_{1}),... ,(x_{N},y_{N})\}}
    where each item x_{i} has an associated class y_{i}\in \{-1,1\},
    and a set of weak classifiers {k_{1},..,k_{L} each of which outputs
    a classification k_{j}(x_{i})\in \{-1,1\} for each item.
    '''
    predictions = np.array([classifier.predict(X) for classifier in self.h_classifiers])
    predictions[predictions == 0] = -1

    return np.dot(predictions.T, self.coefficient_weight) >= 0

  def score(self, X, y):
    y_hat = self.predict(X)
    return sum(y_hat == y) / float(len(y))



