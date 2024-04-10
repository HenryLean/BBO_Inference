import numpy as np
from __utils import Critic

class FourPointInfer(Critic):
    def __init__(self, est:float, var:float, hypara:float=3.):
        super(FourPointInfer, self).__init__(est, var)
        self.N_obs = 0
        self.hypara = hypara
    
    def update(self, obs):
        mu = np.mean(np.concatenate(obs))
        s2 = np.var(np.concatenate(obs), ddof=1)
        b2 = (mu - self.est)**2
        tau = sum(list(map(len, obs)))
        N = self.N_obs + tau
        self.var = self.var + tau * (s2 * (tau - 1) / tau + b2 * self.N_obs / N - self.var) / (N-1)
        est_ = list(map(np.mean, obs))
        est_new = (self.hypara**2 * (est_[0] + est_[1]) - (est_[2] + est_[3])) / 2 / (self.hypara**2 - 1)
        self.est = self.est + tau * (est_new - self.est) / N
        self.N_obs = N

    def reset(self, clear=False, *args, **kwargs):
        if clear:
            super(FourPointInfer, self).__init__(*args, **kwargs)
        self.N_obs = 0