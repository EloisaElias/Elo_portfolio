from DecisionTree_rf import DecisionTree
from numpy.random import choice
from collections import Counter
import numpy as np
import random

class RandomForest(object):
    '''
    Random forest class
    '''
    def __init__(self, num_trees, num_features):
        '''
        num_trees: Number of decision trees to create in the forest
        num_features: The number of features to consider when choosing the best split for each node
        '''
        self.num_trees = num_trees
        self.num_features = num_features
        self.forest = None

    def fit(self, X, y):
        '''
        INPUT:
            - X: 2d numpy array
            - y: 1d numpy array
        OUTPUT: None
        '''
        self.forest = self.build_forest(X, y, self.num_trees, X.shape[0], self.num_features)

    def build_forest(self, X, y, num_trees, num_samples, num_features):
        '''
        Return a list of num_trees DecisionTrees.
        build_forest
        Repeat num_trees times

        **Boostrap aggregating | Bagging**
        Repeat num_trees times:
        Create a random sample of the data with replacement
        Build a decision tree with that sample
        Return the list of the decision trees created
        '''
        forest = [] # DecisionTree list
        for tree in xrange(num_trees):
            indices = [random.randint(0, len(X)-1) for i in xrange(num_samples)]# random indices sample
            dt = DecisionTree(num_features = num_features)
            dt.fit(X[indices], y[indices])
            forest.append(dt) # create list of dt

        return forest

    def predict(self, X):
        '''
        Each Decision Tree classify each data point.
        Choose the label with the majority of trees.
        '''
        # calling method predict to the DecisionTree model as sklearn 'clf.predict(X_test)'
        dtrees_predictions = np.array([dt_clf.predict(X) for dt_clf in self.forest]).T
        label_most_trees = np.array([Counter(row).most_common(1)[0][0] for row in dtrees_predictions])
        return label_most_trees

    def score(self, X, y):
        '''
        Return the Random forest accurary
        '''
        return sum(self.predict(X) == y) / float(len(y))



if __name__ == '__main__':
    from sklearn.cross_validation import train_test_split
    import pandas as pd

    df = pd.read_csv('data/congressional_voting.csv', names=['Party']+range(1, 17))
    y = df.pop('Party').values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    rdf = RandomForest(num_trees=10, num_features=10)
    rdf.fit(X_train, y_train)
    print 'RandomForest score', rdf.score(X_test, y_test)


