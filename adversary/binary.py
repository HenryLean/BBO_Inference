import numpy as np

class BinaryEnv(object):
    def __init__(self, meanFn, *fn, **kw):
        self.mean = meanFn

    def get_obs(self, action, size=1, *args, **kwargs):
        mean = max(0, min(1, self.mean(action)/5))
        rvs = np.random.binomial(1, mean, size=size)
        return rvs
