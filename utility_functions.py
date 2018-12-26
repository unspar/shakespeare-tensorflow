import numpy as np


class ModelFunctions():
  '''
  functions used by the Model class.
  Mostly static methods
  '''
  
  @staticmethod
  def weighted_pick( weights):
    '''
    Takes a vector of weights
    returns an index into weights randomly seleted
    with the value of weight in weights determining the probability
    '''
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(int(np.searchsorted(t, np.random.rand(1)*s)))




