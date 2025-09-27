import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm


class BaseSSI(ABC):
    def __init__(self, est, var, *args, **kwargs):
        self.est = est
        self.var = var
        self.mu = 0
        self.N_obs = 0


    def get_CI(self, level=0.95):
        z = np.array([norm.ppf(1 + i*level)/2 for i in (-1, 1)])
        return self.est + z * np.sqrt(self.var)
    

    def update_vanilla(self, batch, k, *, rate_var=1):
        y_hat = np.mean(batch)
        b1 = (y_hat - self.mu)
        s2 = np.var(batch)
        self.var = self.var + (k/(k+1) * b1**2 + s2 - self.var) / (k+1)**rate_var
        self.mu = self.mu + b1 / (k+1)


    @abstractmethod
    def update(self, y_bar, s2, batch_size):
        pass


    def _get_b2N(self, y_bar, batch_size):
        b2 = (y_bar - self.est)**2
        N = self.N_obs + batch_size
        return b2, N
    

    def _update(self, y_bar, k):
        b = y_bar - self.est
        self.est = self.est + b / (k+1)
        self.var = self.var + (k / (k+1) * b**2 - self.var) / (k+1)



class OrdinSSI(BaseSSI):
    def update(self, y_bar, s2, batch_size):
        b2, N = super()._get_b2N(y_bar, batch_size)
        self.var = self.var + batch_size / (N-1) * (s2 + b2 * self.N_obs / N - self.var)
        self.est = self.est + batch_size * (y_bar - self.est) / N
        self.N_obs = N


    def _update_vanilla(self, y_bar, batch, k):
        self.est = self.est + (y_bar - self.est) / (k+1)
        super().update_vanilla(batch, k)

    

class ConstSSI(BaseSSI):
    def __init__(self, est, var, rate:float=.05, *args, **kwargs):
        super().__init__(est, var)
        self.rate = rate


    def update(self, y_bar, s2, batch_size):
        b2, N = super()._get_b2N(y_bar, batch_size)
        self.var = self.var + batch_size / (N-1) * (s2 + b2 * self.N_obs / N - self.var)
        self.est = self.est + self.rate * (y_bar - self.est)
        self.N_obs = N


    def _update(self, y_bar, k):
        b = y_bar - self.est
        self.est = self.est + b * self.rate
        self.var = self.var + (k / (k+1) * b**2 - self.var) / (k+1)


    def _update_vanilla(self, y_bar, batch, k):
        self.est = self.est + (y_bar - self.est) * self.rate
        super().update_vanilla(batch, k)



class MultiSSI(BaseSSI):
    def __init__(self, est, var, rate_mu, rate_var, *args, **kwargs):
        super().__init__(est, var)
        self.rate_mu, self.rate_var = rate_mu, rate_var


    def update(self, y_bar, s2, batch_size):
        b2, N = super()._get_b2N(y_bar, batch_size)
        self.var = self.var + (s2 + b2 * self.N_obs / N - self.var) * (batch_size / (N-1))**self.rate_var
        self.est = self.est + (y_bar - self.est) * (batch_size / N)**self.rate_mu
        self.N_obs = N


    def _update(self, y_bar, k):
        b = y_bar - self.est
        self.est = self.est + b / (k+1)**self.rate_mu
        self.var = self.var + (k / (k+1) * b**2 - self.var) / (k+1)**self.rate_var

    
    def _update_vanilla(self, y_bar, batch, k):
        self.est = self.est + (y_bar - self.est) / (k+1)**self.rate_mu
        super().update_vanilla(batch, k, rate_var=self.rate_var)
