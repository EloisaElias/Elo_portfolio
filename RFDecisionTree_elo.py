import pandas as pd
import numpy as np
import math
from collections import Counter
from TreeNode_elo import TreeNode


class DecisionTree(object):
    '''
    Decision Tree class
    '''
    def __init__(self, impurity_criterion='entropy', num_features=None):
        self.num_features = num_features  # features to consider is randomly chosen at each node
        self.root= None # root node
        self.features_name = None # string name of the features
        self.categorical = None  # categorical or continous
        self.impurity_criterion = self._entropy if impurity_criterion == 'entropy' else self._gini


    def fit(self, X, y, features_name=None):
        '''
        INPUT:
                - X: 2d numpy array
                - y: 1d numpy array
                - feature_names: numpy array of strings
        OUTPUT: None
        X is a 2 dimensional array with each column being a feature and each
            row a data point.
        y is a 1 dimensional array with each value being the corresponding
            label.
        feature_names is an optional list containing the names of each of the
            features.
        '''
        if features_name is None or len(features_name) != X.shape[1]:
           self.features_name = np.arange(X.shape[1])
        else:
           self.features_name = features_name
        # Create True/False array of whether the variable is categorical
        is_categorical = lambda x: isinstance(x, str) or isinstance(x, bool) or isinstance(x, unicode)
        self.categorical = np.vectorize(is_categorical)(X[0])

        self.root = self._build_tree(X, y)


    def _build_tree(self, X, y):
        '''
        INPUT:  x - 2d array
                y - 1d array
        OUTPUT: Root Node
        ---
        function BuildTree:
        If every item in the dataset is in the same class
        or there is no feature left to split the data:
            return a leaf node with the class label
        Else:
            find the best feature and value to split the data
            split the dataset
            create a node
            for each split
                call BuildTree and add the result as a child of the node
            return node

        '''
        node = TreeNode() # object instatiation
        index, value, splits = self._choose_split_index(X, y)
        # max_index, max_value, _make_split()
        # max_index: (Int)Index of feature(to split)
        # max_value: key with the max value

        if index is None or len(np.unique(y)) == 1:
            node.leaf = True # (bool) True if node == leaf, otherwise False
            node.classes =  Counter(y)
            node.name = node.classes.most_common(1)[0][0]

        else:
            X1, y1, X2, y2 = splits

            node.column  = index # max_index, (Int)Index of feature(to split)
            node.name = self.features_name[index] # fit function
            node.value = value # max_value
            node.categorical = self.categorical[index] # fit function
            node.left = self._build_tree(X1, y1)
            node.right = self._build_tree(X2, y2)
        return node

    def _entropy(self, y):
        '''
        Here there are n classes in the set and Ki is the i-th class of our target y.
        '''
        # entropy --> impurity
        categories = Counter(y)
        entropy = [] # Quantify how much information(or 'uncertainty') i will gain by learning(ask) about y
        for k, v in categories.iteritems():
            pd = v / float(len(y)) # probability distribution
            binary = np.log2(pd) # measurement
            entropy.append(pd * binary)

        return -1*sum(entropy)

    def _gini(self, y):
        '''
        Gini impurity used by CART classification and regression tree
        Gini answers - What is the probability that is it labeled incorrectly?
        '''
        categories = Counter(y)
        gini = []
        for k, v in categories.iteritems():
            pd = v / float(len(y)) # Probability distribution
            gini.append(pd**2.)

        return 1 - sum(gini)

    def _make_split(self, X, y, split_index, split_value):
        '''
        For categorical features this should split on weather it's equal to the value of not.
        For continuous, it should split on < or >=.

        From _choose_split_index:
        split_index: max_index, (Int)Index of feature(to split)
        split_value: # key with the max value

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
            mask = X[:,split_index] == split_value

        else:
            mask = X[:,split_index] >= split_value

        X1 = X[mask]
        y1 = y[mask]
        X2 = X[~mask] # ~ binary complement
        y2 = y[~mask] # ~ binary complement

        return X1, y1, X2, y2


    def _information_gain(self, y, y1, y2):
        '''
        gain = Entropy(parent) - Avg(entropy(children))
        his should take the data and try every possible
        feature and value to split on. It should find the one with the best information gain.
        '''
        H = self._entropy
        gain = H(y) - len(y1)/float(len(y)) * H(y1) - len(y2)/float(len(y)) * H(y2)
        return gain




    def _choose_split_index(self, X, y):
        '''
        INPUT   X: 2d array
                y: 1d array
        OUTPUT  split_index(int) : Feature index
                value(int/float/bool/str) : Feature value
                splits(X1, y1, X2, y2)
        Determine which feature and value to split on.
        Return the index and value of optimal split along with the split of the dataset
        Return None, None, None if there is no split which improves information gain
        call:
        >>> index, value, splits = self._choose_split_index(X, y)
        '''

        # Randomly selection of num_features
        features_ss = np.random.choice(X.shape[1], self.num_features, replace=False) # ss: subset
        X = X[:, features_ss]

        max_gain = 0
        split_index = None # (Int)Index of feature(to split)
        split_value = None
        splits = None

        for j in xrange(X.shape[1]): # j : features (index)
            values = np.unique(X[:, j])
            if len(values) < 2:
                continue # return to initial condition (j)

            for value in values:
                #if X[i][j] not in gain_dict: # X[j][i]: feature value
                temp_splits = self._make_split(X, y, j, value) # x[0,2] = x[0][2]
                X1, y1, X2, y2 = temp_splits
                gain = self._information_gain(y, y1, y2)

                if gain > max_gain:
                    max_gain = gain
                    split_value = value # key with the max value
                    split_index = j # feature index: max_index, (Int)Index of feature(to split)
                    splits = temp_splits

        return split_index, split_value, splits



    def predict(self, X):
        '''
        INPUT: X: 2d array
        OUTPUT: y: 1d array
        Calling predict_one, return the predicted label for a single data point.
        Return an array of predictions for the feature matrix X.
        '''
        return np.apply_along_axis(self.root.predict_one, axis=1, arr=X)


    def __str__(self):
        return str(self.root)












































































