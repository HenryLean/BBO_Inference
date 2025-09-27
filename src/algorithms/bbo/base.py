import numpy as np
from abc import abstractmethod

from src._utils.memory import Memory
from gradopt.gradient_descent import BaseGD



class BaseBBOSSI(BaseGD):
    def __init__(self, init, infer, env, bounds, *args, **kwargs):
        super(BaseBBOSSI, self).__init__(init, np.array(bounds, np.float64), **kwargs)
        self.infer = infer
        self.env = env
        self.memory = Memory()
        try:
            distance = np.linalg.norm(np.array(self.action) - kwargs["optimizer"])
            performance = self.env.get_performance(self.action)
            gap = performance - kwargs["opt_val"]
            self.memory.collect(
                np.array(self.action),
                self.infer.est,
                self.infer.var,
                distance=distance,
                performance=performance,
                regret=abs(gap),
                y_bar=0
            )
        except: pass


    def _get_perturbation(self, sphere:bool=True):
        if sphere:
            unit = np.random.normal(size=self.dim)
            return unit / np.linalg.norm(unit)
        else:
            return np.random.choice([-1,1], size=self.dim)


    def iterate(self, lr, perturb_size, batch_size, sphere:bool=True, *args, **kwargs):
        u = self._get_perturbation(sphere)
        episodes = self.get_episodes(u, perturb_size, batch_size)
        gradient = self.get_gradient(episodes, u, perturb_size)
        self.action = self._projection(super().__call__(gradient, lr=lr, *args, **kwargs))
        return episodes
    

    def run(self, perturb_fn, step_fn, batch_fn, N:int=1000, *, indexes=None, optimizer=0, opt_val=0, est_var_method="iterative", **kwargs):
        k = 0
        while k < N:
            lr = step_fn(k)
            batch_size = batch_fn(k)
            perturb_size = perturb_fn(k)

            episodes = self.iterate(lr, perturb_size, batch_size, **kwargs)

            y_bar = self.update_inference(episodes, k, est_var_method=est_var_method)

            if indexes is None or k in indexes:
                distance = np.linalg.norm(np.array(self.action) - optimizer)
                performance = self.env.get_performance(self.action)
                self.memory.collect(
                    np.array(self.action),
                    self.infer.est,
                    self.infer.var,
                    distance=distance,
                    performance=performance,
                    regret=abs(performance-opt_val),
                    y_bar = y_bar
                )
            k += 1


    @abstractmethod
    def get_episodes(self, perturb_dir, perturb_size, batch_size:int=1):
        pass
    

    @abstractmethod
    def get_gradient(self, episodes, perturb_dir, perturb_size, *args, **kwargs):
        pass


    @abstractmethod
    def update_inference(self, episodes):
        pass


