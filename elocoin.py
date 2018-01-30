import random

class Coin(object):
  def flip_biased(self):
    if random.random() < 0.56:
      return 'H'
    else:
      return 'T'
  def flip(self):
    if random.random() < 0.50:
      return 'H'
    else:
      return 'T'
