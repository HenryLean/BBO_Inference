import numpy as np
from scipy.stats import norm
from .functions import projection


class Optimizer(object):
    def __init__(self, act:np.ndarray, hypara=None):
        """
        x: the initial decision, it would specify dimensions
        """
        self.act = np.array(act, float)
        self.dim = np.shape(act)[0]
        self.hypara = hypara
    
    def update(self, gradient, stepsize, bounds:dict):
        self.act = self.act - gradient * stepsize
        self.act = np.array(list(map(projection, self.act.tolist(), bounds['lb'], bounds['ub'])))


class Critic(object):
    def __init__(self, est:float, var:float):
        self.est = est
        self.var = var

    def get_CI(self, level):
        args = [(1 + i*level)/2 for i in (-1, 1)]
        zBounds = list(map(norm.ppf, args))
        return self.est-zBounds*np.sqrt(self.var), self.est+zBounds*np.sqrt(self.var)