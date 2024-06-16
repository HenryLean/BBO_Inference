import numpy as np
from __utils import Critic


class ConstInfer(Critic):
    def __init__(self, est:float, var:float, rate:float=.05):
        super(ConstInfer, self).__init__(est, var)
        self.N_obs = 0
        self.rate = rate

    def update(self, obs):
        mu = np.mean(np.concatenate(obs))
        s2 = np.var(np.concatenate(obs), ddof=1)
        b2 = (mu - self.est)**2
        tau = sum(list(map(len, obs)))
        N = self.N_obs + tau
        self.var = self.var + tau * (s2 * (tau - 1) / tau + b2 * self.N_obs / N - self.var) / (N-1)
        self.est = self.est + self.rate * (mu - self.est)
        self.N_obs = N

    def reset(self, clear=False, *args, **kwargs):
        if clear:
            super(ConstInfer, self).__init__(*args, **kwargs)
        self.N_obs = 0
