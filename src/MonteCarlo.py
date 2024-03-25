from .__init__ import *
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


class SimPEJ(object):
    def __init__(self, arrivalRate:float=1):
        self.__arrivalRate = arrivalRate

    def rvs(self, t:float, dim:int, sclr:np.ndarray,  size:int=1):
        jump = np.zeros(shape=(size, dim))
        s = -self.__arrivalRate * np.log(np.random.rand(size))
        additivity = (s <= t)
        while sum(additivity):
            D = -sclr * np.log(np.random.uniform(size=(size, dim)))
            jump = jump + D
            s = s -self.__arrivalRate * np.log(np.random.rand(size))
            additivity = (s <= t)
        return jump


class SimCPJ(object):
    discount = False
    def __init__(self, arrivalRate:float=1, rvType:str="const", **kwargs):
        self.__arrivalRate = arrivalRate
        self.__rvGenerator = rvGenerator[rvType](**kwargs)

    def simulate(self, t:float, d:int=1, size:int=1):
        jump = 0
        s = -self.__arrivalRate * np.log(np.random.rand(size))
        additivity = (s <= t)
        while sum(additivity):
            D = (additivity * self.__rvGenerator.rvs(size=(d,size))).T # * np.random.choice([-1,1])
            jump = jump + (D.T * np.exp(s)).T if self.discount else jump + D
            s = s - self.__arrivalRate * np.log(np.random.rand(size))
            additivity = (s <= t)
        return jump
    

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

