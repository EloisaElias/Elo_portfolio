import pandas as pd
from intertools import izip
from DecisionTree_elo import DecisionTree

def test_tree(filename):
  df = pd.read_csv(filename)
  y = df.pop('Result').values
  X = df.values
  print X

  tree = DecisionTree()
  tree.fit(X, y, df.columns)
  print tree

  y_predict = tree.predict(X)
  print 'FEATURE, ACTUAl, PREDICTED'
  print '========, ======, ========'
  for features, true, predicted in izip(X, y, y_predict)
    print '{}, {}, {}'.format(str(features), str(true), str(predicted))


  if __name__ == '__main__':
    test_tree('data/playgolf.csv')
