import numpy as np
from scipy.stats import gamma



class BaseEnv(object):
    def __init__(self, mean, *, epsilon=1e-6, **kwargs):
        self.mean = mean
        self.epsilon = epsilon

    def get_obs(self, action, size=1, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement get_obs.")
    
    def get_performance(self, action):
        action = np.array(action, dtype=float)
        return self.mean(action)


class BinaryEnv(BaseEnv):
    def get_obs(self, action, size=1, *args, **kwargs):
        x = np.array(action, dtype=float)
        mean = max(self.epsilon, min(1 - self.epsilon, self.mean(x)))
        return np.random.binomial(1, mean, size=size)


class ExponentialEnv(BaseEnv):
    def get_obs(self, action, size=1, *args, **kwargs):
        x = np.array(action, dtype=float)
        mean = max(self.epsilon, self.mean(x))
        return -np.log(np.random.uniform(size=size)) * mean


class GammaEnv(BaseEnv):
    def get_obs(self, action, size=1, alpha=4, *args, **kwargs):
        x = np.array(action, dtype=float)
        mean = max(self.epsilon, self.mean(x))
        return gamma.rvs(a=alpha, scale=mean / alpha, size=size)


class GaussianEnv(BaseEnv):
    def __init__(self, mean, noise, *, epsilon=1e-6, **kwargs):
        super().__init__(mean, epsilon=epsilon, **kwargs)
        self.noise = noise

    def get_obs(self, action, size=1, *args, **kwargs):
        x = np.array(action, dtype=float)
        mean = self.mean(x)
        std = self.noise(x)
        return np.random.normal(mean, std, size=size)


class LogNormEnv(BaseEnv):
    def get_obs(self, action, size=1, *args, **kwargs):
        x = np.array(action, dtype=float)
        mean = max(self.epsilon, self.mean(x))
        std = 1  # self.noise(x)
        logs = np.random.normal(loc=np.log(mean) - std * std / 2, scale=std, size=size)
        return np.exp(logs)


class ParetoEnv(BaseEnv):
    def get_obs(self, action, size=1, alpha=3, *args, **kwargs):
        x = np.array(action, dtype=float)
        mean = self.mean(x)
        scale = mean * (alpha - 1) / alpha
        return scale * np.random.uniform(size=size) ** (-1 / alpha)


