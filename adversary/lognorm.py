import numpy as np


class LogNormEnv(object):
    def __init__(self, meanFn, noiseFn, epsilon=1e-6, *args, **kwargs):
        self.mean = meanFn
        self.noise = noiseFn
        self.eps = epsilon

    def get_obs(self, action, size=1, *args, **kwargs):
        x = np.array(action, dtype=float)
        mean = max(self.eps, self.mean(x))
        # mean = self.mean(x)
        std = self.noise(x)
        logs = np.random.normal(loc=np.log(mean)-std*std/2, scale=std, size=size)
        rvs = np.exp(logs)
        return rvs
    