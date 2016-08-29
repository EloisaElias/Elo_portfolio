import numpy as np

def hypothesis(X, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array
    OUTPUT: numpy array

    Calculate the predicted percentages (floats between 0 and 1) for the given
    data with the given coefficients.
    '''
    # print coeffs

    return 1. / (1. + np.exp(-np.dot(X, coeffs.reshape(len(coeffs), 1))))


def predict(X, coeffs):
    '''
    INPUT: 2 dimensional numpy array, numpy array
    OUTPUT: numpy array

    Calculate the predicted values (0 or 1) for the given data with the given
    coefficients.
    '''

    return hypothesis(X, coeffs).round()


def log_likelihood(X, y, coeffs, lbda=1):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array
    OUTPUT: float

    Calculate the log likelihood of the data with the given coefficients.
    '''

    lreg =  np.sum(coeffs**2)
    hx = hypothesis(X, coeffs)
    return np.sum(y * np.log(hx) + (1 - y) * np.log(1-hx) + lbda*lreg)

def log_likelihood_gradient(X, y, coeffs, l=1):
    '''
    INPUT: 2 dimensional numpy array, numpy array, numpy array
    OUTPUT: numpy array

    Calculate the gradient of the log likelihood at the given value for the
    coeffs. Return an array of the same size as the coeffs array.
    '''

    dlreg = 2*coeffs
    y = y.reshape(len(y), 1) # column vector
    dif = (y - hypothesis(X, coeffs)).T

    return np.squeeze(dif.dot(X)) + dlreg

def accuracy(y_true, y_pred):
    '''
    INPUT: numpy array, numpy array
    OUPUT: float

    Calculate the percent of predictions which equal the true values.
    '''
    pass

def precision(y_true, y_pred):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: float

    Calculate the percent of positive predictions which were correct.
    '''
    pass

def recall(y_true, y_pred):
    '''
    INPUT: numpy array, numpy array
    OUTPUT: float

    Calculate the percent of positive cases which were correctly predicted.
    '''
    pass

def add_intercept(X):
    '''
    INPUT: 2 dimensional numpy array
    OUTPUT: 2 dimensional numpy array

    Return a new 2d array with a column of ones added as the first
    column of X.
    '''

    return np.insert(X, 0, 1, axis=1)

def scale_X(X):
    mu = X.mean(axis = 0)
    std_dev = X.std(ddof =1 , axis =0)
    X_scaled = (X-mu)/std_dev
    return X_scaled

