from __future__ import division
from collections import OrderedDict

import itertools
import matplotlib.pyplot as plt


class Dbayes(object):
  # Bayes inference for discrete data
  def __init__(self, prior, likelihood_f):

    self.prior = prior
    self.likelihood_f = likelihood_f
    self.normalize()


  def normalize(self):
    # Evidence : The probability of observing the data.
    # In Bayesian analysis, this term ensures the sum of all probabilities is 1

    prior_p = 1. / float(sum(self.prior.values()))

    for k, v in self.prior.iteritems():
      self.prior[k] = v*prior_p

  def update_posterior(self, data):
    # The product of prior and likelihood (Bayesian-update)
    # The posterior probability becomes the prior of the next Bayesian-update.

    for k, v in self.prior.iteritems():
      likelihood = self.likelihood_f(data, k)
      self.prior[k] = v * self.likelihood_f(data, k)

    self.normalize()

  def get_key(self, key):
    try:
      return int(key)
    except ValueError:
      return key


  def print_distribution(self, label=None):
    # Printing Posterior which is the current sorted Prior
    posterior_p = OrderedDict(sorted(self.prior.items(), key=lambda x: x[0]))
    # ordered_prior_keys : posterior_p : gives you all the distribution lines
    # posterior_p = OrderedDict(sorted(self.prior.items(), key=lambda x: self.get_key(x[0])))
    plt.plot(posterior_p.keys(), posterior_p.values(), alpha=0.6, label=label)
    plt.title('Posterior - Bayes for discrete distributions')
    plt.legend()

    # for k in posterior_p:
    #   print '({},{})'.format(k, self.prior[k])




