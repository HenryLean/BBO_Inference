import numpy as np


class ExponentialEnv(object):
    def __init__(self, meanFn):
        self.mean = meanFn

    def get_obs(self, action, size=1, *args, **kwargs):
        x = np.array(action, dtype=float)
        mean = self.mean(x)
        rvs = -np.log(np.random.uniform(size=size)) * mean
        return rvs