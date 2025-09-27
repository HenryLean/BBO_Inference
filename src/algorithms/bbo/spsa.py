import numpy as np
from .base import BaseBBOSSI



class BaseSPSA(BaseBBOSSI):
    def __init__(self, init, infer, env, bounds, *args, **kwargs):
        super().__init__(init, infer, env, bounds, *args, **kwargs)


    def get_episodes(self, perturb_dir, perturb_size, batch_size:int=1):

        actions_perturbed = [
            self._projection(self.action + i * perturb_dir * perturb_size) for i in (-1, 1)
        ]

        return [self.env.get_obs(a, batch_size) for a in actions_perturbed]


    def get_gradient(self, episodes, perturb_dir, perturb_size, *args, **kwargs):
        return np.mean(episodes[1] - episodes[0]) / (2*perturb_size) * perturb_dir


    def update_inference(self, episodes, k, *, est_var_method="iterative"):
        episodes = np.array(episodes)
        y_bar = np.mean(episodes.flatten())
        if est_var_method == "iterative":
            s2 = np.var(episodes.flatten())
            tau = episodes.size
            self.infer.update(y_bar, s2, tau)
        elif est_var_method == "vanilla":
            batch = np.concatenate(episodes)
            self.infer._update_vanilla(y_bar, batch, k)
        else:
            self.infer._update(y_bar, k)

        return y_bar



class BaseSPSA1s(BaseBBOSSI):
    def __init__(self, init, infer, env, bounds, *args, **kwargs):
        super().__init__(init, infer, env, bounds, *args, **kwargs)


    def get_episodes(self, perturb_dir, perturb_size, batch_size:int=1):

        actions = [self._projection(self.action + perturb_dir * perturb_size), self.action]

        return [self.env.get_obs(a, batch_size) for a in actions]
    

    def get_gradient(self, episodes, perturb_dir, perturb_size, *args, **kwargs):
        return np.mean(episodes[0] - episodes[1]) / (2*perturb_size) * perturb_dir


    def update_inference(self, episodes, k, *, est_var_method="iterative"):
        y_bar = np.mean(episodes[1])
        if est_var_method == "iterative":
            s2 = np.var(episodes[1])
            tau = episodes[1].size
            self.infer.update(y_bar, s2, tau)
        elif est_var_method == "vanilla":
            batch = np.concatenate(episodes)
            self.infer._update_vanilla(y_bar, batch, k)
        else:
            self.infer._update(y_bar, k)
        
        return y_bar
