from __utils import *


class FourPoint(Optimizer):
    def __init__(self, act, bounds:dict, hypara=3, *args, **kwargs):
        """
        x: the initial decision/action
        """
        super(FourPoint, self).__init__(act, hypara=hypara, bounds = bounds, *args, **kwargs)

    def iterate(self, env, pertsize, stepsize, tau, **kwargs):
        # u = get_unitsphere(self.dim)
        u = np.random.choice((-1, 1), size=self.dim)
        obs = [env.get_obs(self.act + i * u * pertsize, tau[0]) for i in (-1,1)]
        obs_ = [env.get_obs(self.act + i * u * pertsize * self.hypara, tau[1]) for i in (-1,1)]
        gradient = (np.mean(obs[1] - obs[0]) * self.hypara**3 - np.mean(obs_[1] - obs_[0])) / self.hypara / (self.hypara**2 - 1) / (2 * pertsize) * u
        self.update(gradient, stepsize, self.bounds, **kwargs)
        return self.act, obs + obs_

