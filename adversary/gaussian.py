import numpy as np


class GaussianEnv(object):
    def __init__(self, meanFn, noiseFn):
        self.mean = meanFn
        self.noise = noiseFn

    def get_obs(self, action, size=1, *args, **kwargs):
        x = np.array(action, dtype=float)
        mean = self.mean(x)
        std = self.noise(x)
        rvs = np.random.normal(mean, std, size=size)
        return rvs
