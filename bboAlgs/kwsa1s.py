from __utils import *
from functools import partial


class KWSA_1side(Optimizer):
    def __init__(self, act, bounds:dict, *args, **kwargs):
        """
        x: the initial decision
        """
        super(KWSA_1side, self).__init__(act, *args, **kwargs)
        self.bounds = bounds
        self.indices = list(range(self.dim))

    def get_perturb_act(self, index, pertsize):
        vec = np.zeros(self.dim)
        vec[index] = 1
        return self.act + vec * pertsize

    def iterate(self, env, pertsize, stepsize, *args):
        partial_by = partial(self.get_perturb_act, pertsize=pertsize)
        perturb_ = list(map(partial_by, self.indices))
        obs = env.get_obs(self.act, size=self.dim)
        obs_p = np.array(list(map(env.get_obs, perturb_)))
        gradient = ((obs_p - obs) / (2 * pertsize))[:,0]
        self.update(gradient, stepsize, self.bounds)
        return self.act, [obs]