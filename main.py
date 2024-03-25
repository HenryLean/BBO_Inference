import numpy as np
import pandas as pd
from src import Replicator


# Globals variables
methods = ['cs', '4p']
estimators = ['x', 'mu', 'var']
L = np.array([
    [2, 0],
    [-1, 2]
])
A = L @ L.T
x = np.array([1, -2])
b = A @ x


def mu_fn(x, c=0): return x @ A @ x / 2 - b @ x + c


def sigmoid(y): return 1 / (1 + np.exp(-y))


# Objective functions
## Continuous
def fn_c1(x, tau=1):
    mu = mu_fn(x, c=20)
    sclr = 1 + np.linalg.norm(x) * np.cos(np.pi * np.linalg.norm(x))
    noise = np.random.normal(size=tau)
    return mu + sclr * np.mean(noise)


def fn_c2(x, tau=1):
    mu = 20 * sigmoid(mu_fn(x)/800 - 2)
    sclr = 1 + np.linalg.norm(x) * np.cos(np.pi * np.linalg.norm(x))
    noise = np.random.normal(size=tau)
    return mu + sclr * np.mean(noise)


## Binary/discrete
def fn_b1(x, tau=1):
    p = min(.99, max(.01, mu_fn(x)/2000 + .3))
    return np.random.binomial(tau, p) / tau


def fn_b2(x, tau=1):
    p = min(.99, max(.01, 2 * sigmoid(mu_fn(x)/600 - 2)))
    return np.random.binomial(tau, p) / tau


fn = {
    'c1': fn_c1,
    'c2': fn_c2,
    'b1': fn_b1,
    'b2': fn_b2
}


# repeat comparsions
def repeatCompare(fn, R, n, param, tau, cstr=np.array([10]*2), sclr=1):
    rslt = {}
    threads = {key: {} for key in methods}
    for key in methods:
        for i in range(5):
            name = '{}#{}#'.format(key,i+1)
            threads[key][name] = Replicator(fn, rslt, key, R, n, param[key], tau, cstr, sclr, name)
    
    for key in methods:
        for name in threads[key]:
            threads[key][name].start()

    for key in methods:
        for name in threads[key]:
            threads[key][name].join()

    return rslt



if __name__ == "__main__":
    R = 25
    n = int(1e5)

    print("Input the environment: cheap or expensive samples ([c]/e)?")
    print("\t::input `c' for 20 samples while `e' for 5 samples at each iteration.")
    input_value = input()
    
    if input_value in ['c', 'e']:
        if input_value == 'c':
            env = 'Chp'
            param = {'cs': .05, '4p': 3} 
            tau = (18, 2)
        else:
            env = 'Exp'
            param = {'cs': .05, '4p': np.sqrt(3/2)} 
            tau = (3, 2)
    else:
        env = 'Chp'
        param = {'cs': .05, '4p': 3} 
        tau = (18, 2)
        print("Oops!!! Out of choices, uses default...")
    
    cstr = np.array([10]*2)

    print("Main program launches...")

    sclr = 100
    for ftp in ['c1', 'c2']:
        cache = repeatCompare(fn[ftp], R, n, param, tau, cstr, sclr)
        for key in methods:
            for each in estimators:
                cache[key][each].to_csv("./output/d2{}#{}_{}_{}.csv".format(env, ftp, key, each), index=False)

    sclr = 1000
    for ftp in ['b1', 'b2']:
        cache = repeatCompare(fn[ftp], R, n, param, tau, cstr, sclr)
        for key in methods:
            for each in estimators:
                cache[key][each].to_csv("./output/d2{}#{}_{}_{}.csv".format(env, ftp, key, each), index=False)