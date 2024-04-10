import csv
from __utils import *
from bboAlgs import *
from inference import *
from adversary import *


class BboInfer(object):
    def __init__(self, actor, critic, *args, **kwargs):
        self.actor = actor(**kwargs) if isinstance(actor(**kwargs), Optimizer) else KWSA_1side(**kwargs)
        self.critic = critic(*args) if isinstance(critic(*args), Critic) else OrdinaryInfer(*args)
        # self.env = env(**fnDict)
        self.memory = Memory()

    def run(self, env, perturb_fn, step_fn, batch_fn, N:int=1000, indices:list=None):
        '''
        batch_fn: batch size function
        n: total resources
        '''
        k = 0
        indexes = indices if isinstance(indices, list) else list(range(N))
        while k < N:
            a = step_fn(k)
            b = batch_fn(k)
            c = perturb_fn(k)
            self.actor.act, obs = self.actor.iterate(env, c, a, b)
            self.critic.update(obs)
            if k in indexes:
                self.memory.decisions.append(tuple(self.actor.act))
                self.memory.estimates.append(self.critic.est)
                self.memory.var_ests.append(self.critic.var)
            k += 1
    
    def write(self, key:str='estimates', output_dir:str='./output/'):
        result = {
            "actions": self.memory.decisions,
            "estimates": self.memory.estimates,
            "variances": self.memory.var_ests
        }
        csv_path = output_dir + "{}.csv".format(key)
        with open(csv_path, "a", newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(result[key])