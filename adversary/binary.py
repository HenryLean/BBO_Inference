import numpy as np

class BinaryEnv(object):
    def __init__(self, meanFn, epsilon=1e-6, *fn, **kw):
        self.mean = meanFn
        self.eps = epsilon

    def get_obs(self, action, size=1, *args, **kwargs):
        mean = max(self.eps, min(1-self.eps, self.mean(action)))
        rvs = np.random.binomial(1, mean, size=size)
        return rvs
