import numpy as np
import Gradient_functions as rf


class GradientAscent(object):

    def __init__(self, cost, gradient, predict_func, fit_intercept=True, scale=False):
        '''
        INPUT: GradientAscent, function, function
        OUTPUT: None

        Initialize class variables. Takes two functions:
        cost: the cost function to be minimized
        gradient: function to calculate the gradient of the cost function
        '''

        self.coeffs = None
        self.cost = cost
        self.gradient = gradient
        self.predict_func = predict_func
        self.fit_intercept = fit_intercept
        self.scale = scale


    def run(self, X, y, alpha=0.0001, num_iterations=10000, step_size=None):
        '''
        INPUT: GradientAscent, 2 dimensional numpy array, numpy array
               float, int
        OUTPUT: None

        Run the gradient ascent algorithm for num_iterations repititions. Use
        the gradient method and the learning rate alpha to update the
        coefficients at each iteration.
        '''

        if self.scale==True:
            means = np.mean(X, axis=0)
            stdev = np.std(X, axis=0)
            X = (X-means)/stdev  #problems if stdev = 0

        if self.fit_intercept==True:
            X = rf.add_intercept(X)

        self.coeffs = np.zeros(X.shape[1])


        if not step_size:
            #Use number of iterations criterion
            for i in xrange(num_iterations):
                self.coeffs = self.coeffs + alpha * self.gradient(X, y, self.coeffs)

                if i%1000==0:
                    print "Cost fn: {0}".format(self.cost(X, y, self.coeffs))

        else:
            old_cost = 0
            new_cost = 1000
            #use step size criterion
            count = 0
            while abs(old_cost - new_cost) > step_size:
                old_cost = new_cost
                self.coeffs = self.coeffs + alpha * self.gradient(X, y, self.coeffs)

                new_cost = self.cost(X, y, self.coeffs)

                count += 1
                if count%100==0:
                    print "New cost: {0}".format(new_cost)


    def predict(self, X):
        '''
        INPUT: GradientAscent, 2 dimensional numpy array
        OUTPUT: numpy array (ints)

        Use the coeffs to compute the prediction for X. Return an array of 0's
        and 1's. Call self.predict_func.
        '''
        if self.fit_intercept==True:
            X = rf.add_intercept(X)

        return self.predict_func(X, self.coeffs)
