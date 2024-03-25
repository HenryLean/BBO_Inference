from .__init__ import *


def repeatPortFn(
        pfn,
        tau: tuple,
        const: bool,
        gamma: float = .05,
        n: int=1000,
        r: int = 30,
        name: str = "cs",
        *args, **kwargs
    ):
    init = np.random.uniform(size=2) * np.pi / 2, 0, 0
    pfo = PortfoliOpt(pfn, init, tau, const)
    cache = {each: [] for each in pfo.paths}
    cltr = {}
    tList = []
    for i in range(r):
        s = time.time()
        temp = pfo.optimize(n, gamma, *args, **kwargs)
        t = time.time()
        for each in cache:
            cache[each].append(pd.Series(temp[each], name="{}#{}".format(name, i+1)))
        print(
            "Replica {}#{} finished, takes {:.2f} ms\n\testimates w={}, mu={:.2f} and var={:.2f}".format(
                name, i+1, 1000*(t-s), get_weights(temp['x'][-1]), temp['mu'][-1], temp['var'][-1]
            )
        )
        tList.append(t - s)
        init = np.random.uniform(size=2) * np.pi / 2, 0, 0
        pfo.__init__(pfn, init, tau, const)
    for each in cache:
        cltr[each] = pd.concat(cache[each], axis=1)
    return cltr, pd.Series(tList, name=name)


def collectRslt(lock, cltr, tLists, name, func, args):
    cache, tList = func(*args)
    with lock:
        cltr[name] = cache
        tLists.append(tList)
        print("collect {:s} using args = {}".format(func.__name__, args))