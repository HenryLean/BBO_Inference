import numpy as np


def get_const(k, const, *args): return const

def get_const4(k, consts, *args): return consts

def get_harmony(x, a, b, c): return a / (x + b)**c

def logit(x): return 1 / (np.exp(-x) + 1)

def quadratic_fn(x, L, b, intercept=0.): 
    x = np.array(x)
    L = np.array(L)
    b = np.array(b)
    Q2 = (x @ L.T @ L @ x) / 2
    Q1 = - b @ x
    return Q2 + Q1 + intercept

def logistic_fn(x, L, b, intercept=0.):
    y = quadratic_fn(x, L, b)
    return logit(y) + intercept

def periodic_fn(x, A, T=1., phi=0., intercept=0.):
    y = np.linalg.norm(x) * np.pi * 2 / T + phi
    z = A * np.sin(y) + intercept
    return z

def get_unitsphere(dim):
    rvs = 2 * np.pi * np.random.uniform(size = dim - 1)
    output = np.ones(dim)
    prod = 1.
    for i in range(dim-1):
        output[i] = output[i] * prod * np.cos(rvs[i])
        prod = prod * np.sin(rvs[i])
    output[dim-1] = output[dim-1] * prod
    return output

def projection(x, lb:float=np.inf, ub:float=np.inf):
    is_num = ((isinstance(lb, float) or isinstance(lb, int)))
    M1 = abs(lb) if is_num else np.inf
    is_num = ((isinstance(ub, float) or isinstance(ub, int)))
    M2 = abs(ub) if is_num else np.inf
    return float(max(min(x, M2), -M1))
