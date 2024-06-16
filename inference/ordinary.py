import numpy as np
from __utils import Critic

class OrdinaryInfer(Critic):
    def __init__(self, est:float, var:float, *args):
        super(OrdinaryInfer, self).__init__(est, var)
        self.N_obs = 0
    
    def update(self, obs, **kwargs):
        mu = np.mean(np.concatenate(obs))
        s2 = np.var(np.concatenate(obs), ddof=1)
        b2 = (mu - self.est)**2
        tau = sum(list(map(len, obs)))
        N = self.N_obs + tau
        self.var = self.var + (s2 * (tau - 1) / tau + b2 * self.N_obs / N - self.var) * tau / (N-1)
        self.est = self.est + (mu - self.est) * tau / N
        self.N_obs = N

    def reset(self, clear=False, *args, **kwargs):
        if clear:
            super(OrdinaryInfer, self).__init__(*args, **kwargs)
        self.N_obs = 0


class MultiTSInfer(Critic):
    def __init__(self, est:float, var:float, hypara, *args):
        super(MultiTSInfer, self).__init__(est, var)
        self.N_obs = 0
        self.hypara = hypara
    
    def update(self, obs, **kwargs):
        mu = np.mean(np.concatenate(obs))
        s2 = np.var(np.concatenate(obs), ddof=1)
        b2 = (mu - self.est)**2
        tau = sum(list(map(len, obs)))
        N = self.N_obs + tau
        self.var = self.var + (s2 * (tau - 1) / tau + b2 * self.N_obs / N - self.var) * (tau / (N-1))**self.hypara[0]
        self.est = self.est + (mu - self.est) * (tau / N)**self.hypara[1]
        self.N_obs = N

    def reset(self, clear=False, *args, **kwargs):
        if clear:
            super(OrdinaryInfer, self).__init__(*args, **kwargs)
        self.N_obs = 0
