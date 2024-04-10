from __utils import *
from functools import partial


class KWSA(Optimizer):
    def __init__(self, act, bounds:dict, *args, **kwargs):
        """
        x: the initial decision
        """
        super(KWSA, self).__init__(act, *args, **kwargs)
        self.bounds = bounds
        self.indices = list(range(self.dim))

    def get_perturb_act(self, index, pertsize):
        vec = np.zeros(self.dim)
        vec[index] = 1
        return self.act + vec * pertsize

    def iterate(self, env, pertsize, stepsize, *args):
        partial_by_plus = partial(self.get_perturb_act, pertsize=pertsize)
        partial_by_minus = partial(self.get_perturb_act, pertsize=-pertsize)

        perturb_plus = list(map(partial_by_plus, self.indices))
        perturb_minus = list(map(partial_by_minus, self.indices))
        
        obs = [np.array(list(map(env.get_obs, perturb_minus))), np.array(list(map(env.get_obs, perturb_plus)))]
        gradient = ((obs[1] - obs[0])/(2*pertsize))[:,0]
        self.update(gradient, stepsize, self.bounds)
        return self.act, obs