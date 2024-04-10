from __utils import *


class SPSA(Optimizer):
    def __init__(self, act, bounds:dict, hypara=False, *args, **kwargs):
        """
        x: the initial decision
        """
        super(SPSA, self).__init__(act, hypara=hypara *args, **kwargs)
        self.bounds = bounds

    def iterate(self, env, pertsize, stepsize, tau):
        u = get_unitsphere(self.dim) if self.hypara else np.random.choice((-1,1), size=self.dim)
        obs = [env.get_obs(self.act + i * u * pertsize, np.sum(tau)) for i in (-1,1)]
        gradient = np.mean(obs[1] - obs[0]) / (2 * pertsize) * u
        self.update(gradient, stepsize, self.bounds)
        return self.act, obs

