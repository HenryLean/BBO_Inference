import numpy as np

def fnPerm(x, b=10, scale=1, intercept=0, *args, **kwds):
    """
    http://www.sfu.ca/~ssurjano/perm0db.html
    domain = [-d, d]
    optimum = 0, optimizer = [1/i for i in range(1,d+1)]
    """
    d = len(x)
    a = np.array(list(range(1, d+1)))
    x = np.array(x, dtype=float)
    y = np.sum([((a + b) @ (x**i - 1/a**i))**2 for i in range(1, d+1)]) / 10**d
    return y * scale + intercept
