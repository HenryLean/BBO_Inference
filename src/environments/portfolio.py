import numpy as np
from scipy.stats import norm
from simenvs.financial_sys import PortfolioNormalReturn, PortfolioJumpReturn



class PortfolioNormalEnv(PortfolioNormalReturn):
    def __init__(self, mean, cov=None, target=1, **kwargs):
        super().__init__(mean, cov)
        self.target = target

    def get_obs(self, action, size=1, *args, **kwargs):
        return super().is_satisficed(action, self.target, size)
    
    def get_performance(self, action):
        w = self.get_weights(action)
        sigma = self.cov_factor @ self.cov_factor.T
        z = (self.target - w @ self.mean) / np.sqrt(w @ sigma @ w)
        return 1- norm.cdf(z)


class PortfolioJumpEnv(PortfolioJumpReturn):
    def __init__(self, mean, cov=None, target=1, time_horizon=1, jump_rate=0.1, jump_type="normal", jump_kwargs=None, **kwargs):
        if jump_kwargs is None:
            jump_kwargs = {}
        super().__init__(mean, cov, time_horizon=time_horizon, jump_rate=jump_rate, jump_type=jump_type, jump_kwargs=jump_kwargs)
        self.target = target

    def get_obs(self, action, size=1, *args, **kwargs):
        return super().is_satisficed(action, self.target, size)
