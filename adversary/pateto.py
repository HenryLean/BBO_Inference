import numpy as np


class ParetoEnv(object):
    def __init__(self, meanFn, *args, **kwargs):
        self.mean = meanFn

    def get_obs(self, action, size=1, beta=3, *args, **kwargs):
        mean = self.mean(action)
        scale = mean * (beta-1)/beta
        rvs = scale * np.random.uniform(size=size)**(-1/beta)
        return rvs
    