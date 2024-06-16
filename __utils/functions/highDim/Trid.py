import numpy as np
def fnTrid(x, scale=1, intercept=0, *args, **kwds):
    """
    http://www.sfu.ca/~ssurjano/trid.html
    domain = [-d**2, d**2]
    optimum = -d*(d+4)*(d-1), optimizer = [(i+1)*(d-i) for i in range(d)]
    """
    d = len(x)
    x = np.array(x, dtype=float)
    y = sum((x - 1)**2) - np.dot(x[:d-1], x[1:])
    return y*scale + intercept