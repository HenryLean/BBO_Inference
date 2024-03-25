from .__init__ import *
from .MonteCarlo import get_unit_sphere


class BBOSSI(object):
    def __init__(self, fn, init=(0,0,0), tau=(1, 1), cstr=None):
        self.fn = fn
        self.x, self.m, self.v = init
        self.tau = tau
        self.cstr = cstr
        self.G = 0
        self.path = {
            'x': [self.x],
            'mu': np.array([self.m]),
            'var': np.array([self.v]),
            'cache': np.array([0])
        }

    def perturb(self, c, u):
        x_p = self.x + c * u
        x_m = self.x - c * u

        y_p = self.fn(x_p, np.sum(self.tau))
        y_m = self.fn(x_m, np.sum(self.tau))

        g = (y_p - y_m) / (2 * c) * u
        m = (y_p + y_m) / 2

        return g, m
    
    def perturb4p(self, h, c, u):
        if h <= 1:
            warn('Illegal value, modified `h` <- 3!')
            h = 3

        x_p = self.x + c * u
        x_m = self.x - c * u
        x_pp = self.x + h * c * u
        x_mm = self.x - h * c * u

        y_p = self.fn(x_p, self.tau[0])
        y_m = self.fn(x_m, self.tau[0])
        y_pp = self.fn(x_pp, self.tau[1])
        y_mm = self.fn(x_mm, self.tau[1])

        g = ((y_p - y_m) * h**3 - (y_pp - y_mm)) / (2 * h * (h**2 - 1) * c) * u
        m = ((y_p + y_m) * h**2 - (y_pp + y_mm)) / (2 * (h**2 - 1))

        return g, m
    
    def project(self, x):
        if self.cstr is not None:
            for i in range(len(x)):
                x[i] = max(min(self.cstr[i], x[i]), -self.cstr[i])
        return x

    def update_4p(self, obs, k, r, sclr=1, zeta=1):
        self.v = self.v + (r* (obs[1] - self.m)**2 - self.v) / (k**zeta)
        self.m = self.m + (obs[1] - self.m) / k
        self.x = self.project(self.x - sclr * obs[0] / k)
        # self.path['x'] = np.append(self.path['x'], [self.x], axis=0)
        self.path['x'].append(tuple(self.x))
        self.path['mu'] = np.append(self.path['mu'], self.m)
        self.path['var'] = np.append(self.path['var'], self.v)

    def update_cs(self, obs, k, gamma, r, sclr=1, zeta=1):
        self.v = self.v + (r* (obs[1] - self.m)**2 - self.v) / (k**zeta)
        self.m = self.m + (obs[1] - self.m) * gamma
        self.x = self.project(self.x - sclr * obs[0] / k)
        # self.path['x'] = np.append(self.path['x'], [self.x], axis=0)
        self.path['x'].append(tuple(self.x))
        self.path['mu'] = np.append(self.path['mu'], self.m)
        self.path['var'] = np.append(self.path['var'], self.v)


class Replicator(threading.Thread):
    def __init__(self, fn, cltr, method, R, n, param, tau=(1, 1), cstr=None, sclr=1, name=None, dim=2):
        self.fn = fn
        self.__d = dim
        super().__init__(target=self.repeatSim, name=name, args=(cltr, method, R, n, param, tau, cstr, sclr))

    def spsasi_cs(self, n, gamma, init=(0,0,0), tau=(1, 1), cstr=None, sclr=1, zeta=1):
        test = BBOSSI(self.fn, init, tau, cstr)

        for k in range(n):
            u = get_unit_sphere(self.__d)
            obs = test.perturb(1/(k+1)**(1/5), u)
            test.update_cs(obs, k+1, gamma, k/(k+1), sclr, zeta)
        return test.path

    def spsasi_4p(self, n, h, init=(0,0,0), tau=(1, 1), cstr=None, sclr=1, zeta=1):
        test = BBOSSI(self.fn, init, tau, cstr)

        for k in range(n):
            u = get_unit_sphere(self.__d)
            obs = test.perturb4p(h, 1/(k+1)**(1/5), u)
            test.update_4p(obs, k+1, k/(k+1), sclr, zeta)
        return test.path

    def repeatSim(self, cltr, method, R, n, param, tau, cstr, sclr):
        estimators = ['x', 'mu', 'var']
        cltr[method] = {key: pd.DataFrame() for key in estimators}
        rsltLists = {key: [] for key in estimators}
        spsa = {'cs': self.spsasi_cs, '4p': self.spsasi_4p}

        for i in range(R):
            init = np.random.rand(2) * 20 - 10, 0, 0
            t0 = time.time()
            cache = spsa[method](n, param, init, tau, cstr, sclr)
            t1 = time.time()
            for key in cltr[method]:
                rsltLists[key].append(pd.Series(cache[key], name='{}{}'.format(self.name, i+1)))
            print('Thread {:s} finishes replication {:d} and takes {:.2f} ms\n\tfinal estimation x={}, mu={:.2f} and var={:.2f}'.format(self.name, i+1, 1000*(t1-t0), cache['x'][-1], cache['mu'][-1], cache['var'][-1]))
        for key in cltr[method]:
            temp = pd.concat(rsltLists[key], axis=1)
            cltr[method][key] = pd.concat([cltr[method][key], temp], axis=1)