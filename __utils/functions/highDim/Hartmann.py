import numpy as np

c = np.array([1., 1.2, 3., 3.2])

def fnHartmann(x, *args, **kwds):
    """
    domain = [0,1]
    http://www.sfu.ca/~ssurjano/hart3.html
    optimum = -3.86278, optimizer = (0.114614, 0.555649, 0.852547) if dim==3
    http://www.sfu.ca/~ssurjano/hart6.html
    optimum = -3.32237, optimizer = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573) if dim==6
    """
    x = np.array(x, dtype=float)
    if len(x) == 3:
        A = np.array([
            [3., 10, 30],
            [.1, 10, 35],
            [3., 10, 30],
            [.1, 10, 35]
        ])
        P = np.array([
            [.3689, .1170, .2673],
            [.4699, .4387, .7470],
            [.1091, .8732, .5547],
            [.0381, .5743, .8828]
        ])
    elif len(x) == 6:
        A = np.array([
            [10, 3, 17, 3.5, 1.7, 8],
            [.05, 10, 17, .1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, .05, 10, .1, 14]
        ])
        P = np.array([
            [.1312, .1696, .5569, .0124, .8283, .5886],
            [.2329, .4135, .8307, .3736, .1004, .9991],
            [.2348, .1451, .3522, .2883, .3047, .6650],
            [.4047, .8828, .8732, .5743, .1091, .0381]
        ])
    try:
        return -c @ np.exp(-np.sum(A * (x - P)**2, axis=1))
    except: pass