from src.__init__ import *
from scipy.stats import binom, expon, norm, gamma


def get_unit_sphere(p=2):
    r = 2 * np.pi * np.random.uniform(size = p - 1)
    o = np.zeros(p) + 1
    prod = 1
    for i in range(p-1):
        o[i] = o[i] * prod * np.cos(r[i])
        prod = prod * np.sin(r[i])
    o[p-1] = o[p-1] * prod
    return o


class const(object):
    def __init__(self, const:float=1):
        self.const = const

    def rvs(self, size):
        return np.zeros(size) + self.const


rvGenerator = {
    "const": const,
    "binary": binom,
    "expon": expon,
    "normal": norm,
    "gamma": gamma
}


class Sim0MJ(object):
    def __init__(self, arrivalRate:float=1.0):
        self.__arrival = 1/arrivalRate if arrivalRate > 0 else 1.0

    def rvs(self, t:float, dim:int, sclr:np.ndarray, size:int=1):
        jump = np.zeros(shape=(size, dim))
        s = -self.__arrival * np.log(np.random.rand(size))
        # s = -self.__arrivalRate * np.log(np.random.rand(size, dim))  # element-wise independent
        additivity = (s <= t)
        while np.mean(additivity):
            D = -sclr * np.log(np.random.uniform(size=(size, dim)))
            jump = jump + (D.T * additivity).T
            # jump = jump + D * additivity  # element-wise independent
            s = (s.T -self.__arrival * np.log(np.random.rand(size))).T
            additivity = (s <= t)
        return jump #- sclr * self.__arrival


class SimCPJ(object):
    """
    Monte Carlo jumps: simulation of compound Poisson processes.
    """
    discount = False
    def __init__(self, arrivalRate:float=1, rvType:str="const", **kwargs):
        self.__arrivalRate = arrivalRate
        self.__rvGenerator = rvGenerator[rvType](**kwargs)

    def rvs(self, t:float, d:int=1, sclr:np.ndarray=1,  size:int=1):
        jump = 0
        s = -self.__arrivalRate * np.log(np.random.rand(size))
        additivity = (s <= t)
        while sum(additivity):
            D = (additivity * self.__rvGenerator.rvs(size=(d,size))).T # * np.random.choice([-1,1])
            jump = jump + (D.T * np.exp(s)).T if self.discount else jump + D
            s = s - self.__arrivalRate * np.log(np.random.rand(size))
            additivity = (s <= t)
        return jump


class Portfolio(object):
    checkParam = True
    weight = .5 + np.zeros(2)
    def __init__(self, mu=np.zeros(2), sigma=np.eye(2), w=[], jump:list=[False], **kwargs):
        self.__mu = np.array(mu)
        self.__sigma = np.array(sigma)
        self.checkParam = (len(self.__mu) != self.__sigma.shape[0]) or (len(self.__mu)!= self.__sigma.shape[1])
        if self.checkParam:
            print("!!!?")
            self.__mu = np.zeros(2)
            self.__sigma = np.eye(2)
            print("Assign values by default...")
        self.__dim = len(self.__mu)
        self.__jump = [False] * self.__dim

        if len(w) == self.__dim:
            self.weight = np.array(w)
        else:
            if self.__dim != 2:
                self.weight = 1/self.__dim + np.zeros(self.__dim)
            print("Assign weights by default...")
        self.__jump = jump
        self.rvJump = Sim0MJ(**kwargs)
        # self.rvJump = SimCPJ(**kwargs)

    def checkInit(self):
        print("mu = {}".format(self.__mu))
        print("sigma = {}".format(self.__sigma))
        print("dim = {}".format(self.__dim))
        print("jump = {}".format(self.__jump))
        print("weights = {}".format(self.weight))

    def initWeights(self):
        self.weight = 1/self.__dim + np.zeros(self.__dim)
        print("Initialize weights evenly...")

    def simulate(self, size:int=1, t:float=1):
        t = abs(t)
        rvBase = np.random.normal(size=(self.__dim, size))
        rvBase = t * self.__mu.T + np.sqrt(t) * (self.__sigma @ rvBase).T
        rvJump = np.zeros(shape=(size, self.__dim))
        if self.__jump:
            sclr = np.zeros(self.__dim)
            sclr[self.__jump] = sclr[self.__jump] + self.weight[self.__jump].copy()
            rvJump = self.rvJump.rvs(t, self.__dim, sclr, size)
        return rvBase + rvJump
    

class SimJDM(object):
    def __init__(self, mu:Any, sigma:Any, timeInterval:float, *args, **kwargs):
        self.mu = np.array([mu])
        self.sigma = np.array([sigma])
        self.timeInterval = timeInterval
        self.rvJump = SimCPJ(*args, **kwargs)
        try: 
            dim = self.mu.shape[1]
        except:
            dim = 1
        self.dim = dim

    def checkInit(self):
        print(
            "mu = {},\n".format(self.mu),
            "sigma = {},\n".format(self.sigma),
            "interval = {},\n".format(self.timeInterval),
            "dimension = {},\n".format(self.dim)
        )

    def simulate(self, size:int=1):
        rv = np.random.normal(size=(size,self.dim))
        cache = self.mu * self.timeInterval + rv @ self.sigma
        jumps = self.rvJump.simulate(self.timeInterval, self.dim, size=size)
        path = cache * self.timeInterval + jumps
        return path[0]


class SimOUP(SimJDM):
    def __init__(self, mu:Any, sigma:Any, timeInterval:float, jump:bool=False, *args, **kwargs):
        super().__init__(mu, sigma, timeInterval, *args, **kwargs)
        self.jump = jump
        self.rvJump.discount = True

    def __proceed(self, precedor:Iterable):
        cache = [
            precedor - self.mu,
            np.sqrt((np.exp(2*self.timeInterval) -1)/2) * self.sigma @ np.random.normal(size=self.dim),
            self.rvJump.simulate(self.timeInterval, d=self.dim) if self.jump else 0
        ]
        return self.mu + sum(cache) * np.exp(-self.timeInterval)

    def simulate(self, size:int, init:Iterable=[0]):
        path = np.array(init)
        for _ in range(size):
            obsv = self.__proceed(path[-1])
            path = np.append(path, obsv, axis=0)
        return np.array(path)
