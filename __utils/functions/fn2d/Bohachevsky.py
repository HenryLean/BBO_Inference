from math import cos, pi

def fnBohachevsky(x, Type=0, *args, **kwds):
    """
    see http://www.sfu.ca/~ssurjano/boha.html
    dimension: 2
    region = [-100, 100]
    optimum = 0, optimizer = (0, 0)
    """
    y = x[0]**2 + 2*x[1]**2
    if Type == 0:
        y = y - .3*cos(3*pi*x[0]) - .4*cos(4*pi*x[1]) + .7
    elif Type == 1:
        y  = y - .3*cos(3*pi*x[0])*cos(4*pi*x[1]) + .3
    else:
        y  = y - .3*cos(3*pi*x[0] + 4*pi*x[1]) + .3
    return y