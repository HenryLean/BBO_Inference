import numpy as np
from .functions import projection
from scipy.stats import norm


class Optimizer(object):
    """
    Gradient-based optimization algorithms
    """
    def __init__(self, act:np.ndarray, hypara=None, **kwargs):
        self.act = np.array(act, float)
        self.dim = np.shape(act)[0]
        self.hypara = hypara
        if "bounds" in kwargs.keys():
            self.bounds = kwargs["bounds"]
        else:
            self.bounds = {
                "lb": [np.inf]*self.dim,
                "ub": [np.inf]*self.dim
            }
    
    def update(self, gradient, stepsize, bounds:dict, descent=True):
        self.act = self.act - gradient * stepsize if descent else self.act + gradient * stepsize
        self.act = np.array(list(map(projection, self.act.tolist(), bounds['lb'], bounds['ub'])))

    def random_x0(self, *args, **kwargs):
        u = np.random.uniform(size=self.dim)
        a = np.array(self.bounds["lb"], float)
        b = np.array(self.bounds["ub"], float)
        self.act = a*(1-u) + b*u


class Critic(object):
    def __init__(self, est:float, var:float):
        self.est = est
        self.var = var

    def get_CI(self, level):
        args = [(1 + i*level)/2 for i in (-1, 1)]
        zBounds = list(map(norm.ppf, args))
        return self.est-zBounds*np.sqrt(self.var), self.est+zBounds*np.sqrt(self.var)


class SgdBase(object):
    """
    Gradient-based optimization algorithms
    """
    def __init__(self, act:np.ndarray, **attrs):
        self.act = np.array(act, float)
        self.dim = np.shape(act)[0]
        self.attr = attrs
        if "bounds" not in attrs.keys():
            self.attr["bounds"] = {"lb":[np.inf]*self.dim, "ub":[np.inf]*self.dim}

    def update(self, stepsize, **kwargs):
        gradient = self.get_gradient(**kwargs)
        self.act = self.act - gradient * stepsize
        self.projection()
        return self.act

    def get_gradient(self, estimator, adversary):
        obs = adversary.get_obs(self.act)
        return estimator.get_est(obs)
    
    def projection(self):
        self.act = np.array(list(map(projection, self.act.tolist(), self.attr["bounds"]['lb'], self.attr["bounds"]['ub'])))


class GradientCalBase(object):
    temp = 0
    def __init__(self, init=0, mode=0):
        """
        mode = 0: Waiting time, 1: Sojourn time.
        """
        self.value = init
        self.mode = mode

    def cumulate(self, phi, mul):
        phi = 0 if np.isnan(phi) else float(phi)
        if self.mode == 1:
            self.value = self.value * mul + phi 
        elif self.mode == 0:
            self.value = (self.value + self.temp) * mul
            self.temp = phi
        else:
            self.value = (self.value + phi)*mul
        return self.value