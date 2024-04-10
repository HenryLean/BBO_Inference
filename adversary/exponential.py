import numpy as np


class ExponentialEnv(object):
    def __init__(self, meanFn):
        self.rate = meanFn

    def get_obs(self, action, size=1, *args, **kwargs):
        x = np.array(action, dtype=float)
        mean = 1/self.rate(x)
        rvs = -np.log(np.random.uniform(size=size)) * mean
        return rvs