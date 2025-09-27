import numpy as np
from .base import BaseBBOSSI



class Base4Point(BaseBBOSSI):
    def __init__(self, init, infer, env, bounds, hyperparam=3, *args, **kwargs):
        super().__init__(init, infer, env, bounds, *args, **kwargs)
        self.hyperparam = hyperparam


    def get_episodes(self, perturb_dir, perturb_size, batch_size:int=1):
        actions_perturbed = [
            self._projection(self.action + i * perturb_dir * perturb_size) for i in (-1, 1)
        ] + [
            self._projection(self.action + self.hyperparam * i * perturb_dir * perturb_size) for i in (-1, 1)
        ]
        batch_size_ = int(batch_size / (1+ self.hyperparam**2))
        batch_size_ = 1 if batch_size_ == 0 else batch_size_
        batch_sizes = [batch_size - batch_size_]*2 + [batch_size_]*2


        return [self.env.get_obs(a, b) for a, b in list(zip(actions_perturbed, batch_sizes))]
    

    def get_gradient(self, episodes, perturb_dir, perturb_size, *args, **kwargs):
        D1 = np.mean(episodes[1] - episodes[0])
        D2 = np.mean(episodes[3] - episodes[2])
        return (D1 * self.hyperparam**3 - D2) / (2*perturb_size) * perturb_dir / self.hyperparam / (self.hyperparam**2 - 1)


    def update_inference(self, episodes, k, *, est_var_method="iterative"):
        y1 = np.mean(episodes[0] + episodes[1]) / 2
        y2 = np.mean(episodes[2] + episodes[3]) / 2
        y_bar = (y1 * self.hyperparam**2 - y2) / (self.hyperparam**2 - 1)
        if est_var_method == "iterative":
            s2 = np.var(np.concatenate(episodes))
            tau = np.sum([e.size for e in episodes])
            self.infer.update(y_bar, s2, tau)
        elif est_var_method == "vanilla":
            batch = np.concatenate(episodes)
            self.infer._update_vanilla(y_bar, batch, k)
        else:
            self.infer._update(y_bar, k)
        
        return y_bar