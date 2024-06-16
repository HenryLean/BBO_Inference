from scipy.stats import gamma


class GammaEnv(object):
    def __init__(self, meanFn, epsilon=1e-6, *args, **kwargs):
        self.mean = meanFn
        self.eps = epsilon

    def get_obs(self, action, size=1, alpha=3, *args, **kwargs):
        mean = max(self.eps,self.mean(action))
        rvs = gamma.rvs(a=alpha, scale=mean/alpha, size=size)
        return rvs
    