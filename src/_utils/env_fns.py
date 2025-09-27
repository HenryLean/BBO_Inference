import numpy as np


def get_param(w):
    w = np.array(w).flatten()
    num_negative = np.sum(w<0)
    if num_negative > 0:
        w = np.abs(w)

    if np.sum(w) != 1:
        w = w / sum(w)

    d = w.size-1
    theta = np.zeros(shape=d)
    for i in range(d):
        theta[i] = np.arccos(np.sqrt(w[i]))
        w[i+1:] = w[i+1:] / np.sin(theta[i])**2

    return theta

