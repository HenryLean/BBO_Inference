import numpy as np

def fnZakharov(x, *args, **kwds):
    """
    http://www.sfu.ca/~ssurjano/zakharov.html
    domain = [-5, 10]
    optimum = 0, optimiser = [0 for i in range(d)]
    """
    d = len(x)
    x = np.array(x, dtype=float)
    a = np.array(list(range(1, d+1)))
    return np.sum(x*x) + np.sum(a*x/2)**2 + np.sum(a*x/2)**4