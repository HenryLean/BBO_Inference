from ..__init__ import *
from ..MonteCarlo import Portfolio, get_unit_sphere


def hDefault(*args):
    h = None
    for arg in args:
        if isinstance(arg, float) or isinstance(arg, int):
            h = arg
            break
    if (not h) or (h <= 1):
        h = 3
        warn("Irrational (or no) argument(s) input, set default = 3...")
    return h


def get_weights(x:np.ndarray):
    weights = []
    try:
        d = len(x)
        w = 1
        for i in range(d):
            weights.append(w*np.cos(x[i])**2)
            w = w * np.sin(x[i])**2
        weights.append(w)
    except:
        w = np.cos(x)**2
        weights = [w, 1-w]
    return np.array(weights)


def getRewards(theta:np.ndarray, portfolio:Portfolio, tau=1):
    portfolio.weight = get_weights(theta)
    rvs = portfolio.simulate(tau) @ portfolio.weight
    return rvs


class BlackBoxOpt(ABC):
    def __init__(self, fn, init:tuple, tau:tuple=(1, 1), const:bool=True):
        self.fn = fn
        self.x, self.m, self.v = init
        self.dim = len(init[0])
        self.tau = tau
        self.cs = const
        self.paths = {
            'x': [tuple(self.x)],
            'mu': np.array([self.m]),
            'var': np.array([self.v])
        }

    def perturb(self, c, u, *args):
        x_p, x_m = self.x + c * u, self.x - c * u
        if self.cs:
            y_p, y_m = self.fn(x_p, sum(self.tau)), self.fn(x_m, sum(self.tau))
            g, m = (y_p - y_m) / (2 * c) * u, (y_p + y_m) / 2
        else:
            h = hDefault(*args)
            x_pp, x_mm = self.x + h * c * u, self.x - h * c * u
            y_p, y_m = self.fn(x_p, self.tau[0]), self.fn(x_m, self.tau[0])
            y_pp, y_mm = self.fn(x_pp, self.tau[1]), self.fn(x_mm, self.tau[1])
            g = ((y_p - y_m) * h**3 - (y_pp - y_mm)) / (2 * h * (h**2 - 1) * c) * u
            m = ((y_p + y_m) * h**2 - (y_pp + y_mm)) / (2 * (h**2 - 1))
        return g, m

    def update(self, obs, k, gamma, r=1, sclr=1, zeta=1):
        self.v = self.v + (r* (obs[1] - self.m)**2 - self.v) / (k**zeta)
        if self.cs:
            self.m = self.m + (obs[1] - self.m) * gamma
        else:
           self.m = self.m + (obs[1] - self.m) / k
        self.x = self.project(self.x - sclr * obs[0] / k)

        self.paths['x'].append(tuple(self.x))
        self.paths['mu'] = np.append(self.paths['mu'], self.m)
        self.paths['var'] = np.append(self.paths['var'], self.v)
    
    @abstractmethod
    def project(self, x, *args):
        return x


class PortfoliOpt(BlackBoxOpt):
    cstr = np.pi/2
    def __init__(self, fn, init: tuple, tau: tuple, const: bool = True):
        super().__init__(fn, init, tau, const)

    def project(self, x, *args):
        x_ = x.copy()
        for i in range(len(x_)):
            x_[i] = max(0,min(self.cstr, x_[i]))
        return super().project(x_, *args)
    
    def optimize(self, n: int = 1000, gamma: float=.05, *args, **kwargs):
        for k in range(n):
            u = get_unit_sphere(self.dim)
            obs = self.perturb(1/(k+1)**(1/5), u, *args)
            self.update(obs, k+1, gamma, k/(k+1), **kwargs)
        return self.paths.copy()