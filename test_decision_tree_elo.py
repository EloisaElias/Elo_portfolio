import nose.tools as n
import numpy as np
from DecisionTree_elo import DecisionTree as DT
from TreeNode_elo import TreeNode as TN


def test_entropy():
    array = [1, 1, 2, 1, 2]
    result = DT()._entropy(np.array(array))
    actual = 0.97095
    message = 'Entropy value for %r: Got %.2f. Should be %.2f' \
              % (array, result, actual)
    n.assert_almost_equal(result, actual, 4, message)


def test_gini():
    array = [1, 1, 2, 1, 2]
    result = DT()._gini(np.array(array))
    actual = 0.48
    message = 'Gini value for {}: Got {}. Should be {}'.format(array, result, actual)
    n.assert_almost_equal(result, actual, 4, message)


def fake_data():
    X = np.array([[1, 'bat'], [2, 'cat'], [2, 'rat'], [3, 'bat'], [3, 'bat']])
    y = np.array([1, 0, 1, 0, 1])
    X1 = np.array([[1, 'bat'], [3, 'bat'], [3, 'bat']])
    y1 = np.array([1, 0, 1])
    X2 = np.array([[2, 'cat'], [2, 'rat']])
    y2 = np.array([0, 1])
    return X, y, X1, y1, X2, y2


def test_make_split():
    X, y, X1, y1, X2, y2 = fake_data()
    split_index, split_value = 1, 'bat'
    dt = DT()
    dt.categorical = np.array([False, True])
    result = dt._make_split(X, y, split_index, split_value)
    try:
        X1_result, y1_result, X2_result, y2_result = result
    except ValueError:
        n.assert_true(False, 'result not in correct form: (X1, y1, X2, y2)')
    actual = (X1, y1, X2, y2)
    message = '_make_split got results\n%r\nShould be\n%r' % (result, actual)
    n.ok_(np.array_equal(X1, X1_result), message)
    n.ok_(np.array_equal(y1, y1_result), message)
    n.ok_(np.array_equal(X2, X2_result), message)
    n.ok_(np.array_equal(y2, y2_result), message)


def test_information_gain():
    X, y, X1, y1, X2, y2 = fake_data()
    result = DT()._information_gain(y, y1, y2)
    actual = 0.019973
    message = 'Information gain for:\n%r, %r, %r:\nGot %.3f. Should be %.3f' \
              % (y, y1, y2, result, actual)
    n.assert_almost_equal(result, actual, 4, message)


def test_choose_split_index():
    X, y, X1, y1, X2, y2 = fake_data()
    index, value = 1, 'cat'
    dt = DT()
    dt.categorical = np.array([False, True])
    result = dt._choose_split_index(X, y)
    try:
        split_index, split_value, splits = result
    except ValueError:
        message = 'result not in correct form. Should be:\n' \
                  '    split_index, split_value, splits'
        n.assert_true(False, message)
    message = 'choose split for data:\n%r\n%r\n' \
              'split index, split value should be: %r, %r\n' \
              'not: %r, %r' \
              % (X, y, index, value, split_index, split_value)
    n.eq_(split_index, index, message)
    n.eq_(split_value, value, message)

def test_predict():
    root = TN()
    root.column = 1
    root.name = 'column 1'
    root.value = 'bat'
    root.left = TN()
    root.left.leaf = True
    root.left.name = "one"
    root.right = TN()
    root.right.leaf = True
    root.right.name = "two"
    data = [10, 'cat']
    result = root.predict_one(data)
    actual = "two"
    message = 'Predicted %r. Should be %r.\nTree:\n%r\ndata:\n%r' \
              % (result, actual, root, data)
    n.eq_(result, actual, message)


