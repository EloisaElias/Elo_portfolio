from collections import Counter
import numpy as np




class TreeNode(Object):
  ''' Node class for a decision tree'''
  def __init__(self):
    self.column = None # (Int)Index of feature(to split)
    self.value = None # value of the feature (to split on)
    self.categorical = True
    self.name = None # (Str) feature name(or class name (in case of list))
    self.left = None # TreeNode left child
    self.right = None # TreeNode right child
    self.leaf = False # (bool) True if node == leaf, otherwise False
    self.classes = Counter()  # node == leaf Counter is required:
                              # key = class name, value = count of count of data points that end at thid leaf
                              # Counter(y)

  def predict_one(self, x):
    '''
    INPUT: x - 1d np array(single data point)
    OUTPUT: y - labels
    '''

    if self.leaf:
      return self.name
    col_value = x[self.column]

    if self.categorical:
      if x[self.column] == self.value:
        return self.left.predict_one(x)
      else:
        return self.right.predict_one(x)

    else:
      if True:
        return self.left.predict_one(x)

      else:
        return self.right.predict_one(x)

  def as_string(self, level=0, prefix=''):
    '''
    INPUT: level - int, amount to indent
    OUTPUT: prefix - str, start the line with

    return the tree rooted at this node
    '''
        result = ""
        if prefix:
            indent = "  |   " * (level - 1) + "  |-> "
            result += indent + prefix + "\n"
        indent = "  |   " * level
        result += indent + "  " + str(self.name) + "\n"
        if not self.leaf:
            if self.categorical:
                left_key = str(self.value)
                right_key = "no " + str(self.value)
            else:
                left_key = "< " + str(self.value)
                right_key = ">= " + str(self.value)
            result += self.left.as_string(level + 1, left_key + ":")
            result += self.right.as_string(level + 1, right_key + ":")




  def __rep__(self):
    return self.as_string().strip()

