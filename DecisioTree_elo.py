import pandas as pd
import numpy as np
import math
from collections import Counter, defaultdict
from TreeNode_elo import TreeNode


class DecisionTree(object):
  def __init__(self, impurity_criterion='entropy'):
    self.root= None # root node
    self.features_name = None # string name of the features
    self.categorical = None  # categorical or continous
    self.impurity_criterion = self._h_entropy if impurity_criterion == 'entropy' else self.gini


  def fit(self, X, y, features_name=None):
    if features_name is None or len(features_name) != X.shape[1]:
      self.features_name = np.arange(X.shape[1])
    else:
      self.features_name = features_name

    is_categorical = lambda x: isinstance(x, str) or isinstance(x, bool) or isinstance(x, unicode)

    self.root = self._build_tree(X, y)


  def _build_tree(self, X, y):
    '''
    INPUT:  x - 2d array
            y - 1d array

    OUTPUT: Root Node
    '''
    node = TreeNode() # object instatiation
    index, value, splits = self._chose_split_index(X, y)
    # max_index, max_value, _make_split()

    if index is None or len(np.unique(y)) == 1:
      node.leaf = True
      node.classes =  Counter(y)
      node.name = node.classes.most_common(1)[0][0]

    else:
      X1, y1, X2, y2 = splits
      node.column  = index # max_index
      node.name = self.features_name[index]
      node.value = value # max_value
      node.categorical = self.categorical[index]
      node.left = self._build_tree(X1, y1)
      node.right = self._build_tree(X2, y2)

    return node

  def _h_entropy(self, y):
    # entropy --> impurity
    categories = Counter(y)
    entropy = [] # Quantify how much information(or 'uncertainty') i will gain by learning(ask) about y
    for k, v in categories.iteritems():
      p_y = v / float(len(y)) # probability distribution
      binary = np.log2(p_y)
      entropy.append(p_y * binary)

    return -1*sum(entropy)

  def _gini():
    '''
    Gini impurity used by CART classification and regression tree
    Gini answers - What is the probability that is it labeled incorrectly?
    '''
    categories = Counter(y)
    gini = []
    for k, v in categories.iteritems():
      p_y = v / len(y)
      gini.append(p_y)

    return 1 - sum(gini)

  def _information_gain_hs(self, y, y1, y2):
    '''
    gain = Entropy(parent) - Avg(entropy(children))
    his should take the data and try every possible
    feature and value to split on. It should find the one with the best information gain.
    '''
    h_ = self._h_entropy
    gain_y = h_(y) - len(y1)/float(len(y)) * h_(y1) - len(y2)/float(len(y)) * h_(y2)
    return gain_y


  def _make_split(self, X, y, split_index, split_value):
    '''
    INPUT:  X - 2d array
            y - 1d array
            split_index: int(feature index)
            split_value: int/float/bool/str (feature value)

    OUTPUT: X1 - 2d array
            y1 - 1d array
            X2 - 2d array(complement)
            y2 - 1d array(complement)

    call method: X1, y1, X2, y2 = self._make_split(X, y, split_index, split_value)
    '''
    if self.categorical[split_index]:
      mask = X[:, split_index] == split_value

    else:
      mask = X[:, split_index] >= split_value

    X1 = X[mask]
    y1 = y[mask]
    X2 = X[~mask] # ~ binary complement
    y2 = y[~mask] # ~ binary complement

    return X1, y1, X2, y2



  def _choose_split_index(self, X, y):
    '''
    take the data and try every possible feature and value to split on.
    It should find the one with the best information gain.
    '''
    max_gain = 0
    max_value = None
    max_index = None

    for i in range(X.shape[1]): # obtaining features (index)

      gain_dict = {}
      for j in range(len(X)): # obtain feature value X[j][i]
        if X[j][i] not in gain_dict:
          X1, y1, X2, y2 = self._make_split(X, y, i, X[j][i]) # x[0,2] = x[0][2]
          gain = self._information_gain_hs(y, y1, y2)
          gain_dict[X[j][i]] = gain
      best_split = max(gain_dict, key= lambda K: gain_dict[K]) # best information gain

      if gain_dict[best_split] > max_gain:
        max_gain = gain_dict[best_split]
        max_value = best_split # key with the max value
        max_index = i # feature index

    if max_gain == 0:
      return None, None, None

    return max_index, max_value, self._make_split(X, y, max_index, max_value)




  def predict(self, X):
    # calling predict_one
    return np.apply_along_axis(self.root.predict_one, axis=1, arr=X)


  def __str__(self):
    return str(self.root)












































































