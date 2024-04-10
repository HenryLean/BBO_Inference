import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def cvg_plot(df, true, clevel=.9, tail=0, logIndex=True, plot_dir="./output/test/csv/figs/", *args, **kwargs):
    mean = df.mean()[tail:]
    std = df.std()[tail:]
    alpha = (1 - clevel)/2
    z = norm.ppf(alpha)
    q_l = df.quantile(alpha)[tail:]
    q_u = df.quantile(1-alpha)[tail:]

    indices = np.log10(1+mean.index) if logIndex else mean.index
    plt.figure(**kwargs)
    plt.plot(indices, mean, label="mean estimate")
    plt.axhline(true, color="red", linestyle=":", label="true value")
    plt.fill_between(indices, q_l, q_u, alpha=.3, label="empirical CI")
    plt.fill_between(indices, mean-std*z, mean+std*z, alpha=.2, label="asymptotic CI")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_dir)


def cit_plot(stats, plot_dir, bins=15, *args, **kwargs):
    """
    histogram of t-statistics
    """
    plt.figure(**kwargs)
    xx = np.linspace(min(stats), max(stats), 100)
    yy = norm.pdf(xx)
    plt.hist(stats, density=True, bins=bins, label="histogram")
    plt.plot(xx, yy, linestyle="-.", label="std. normal")
    plt.axvline(0, color="red", linestyle=":")
    plt.grid(True)
    plt.legend()
    plt.savefig(plot_dir)


def __get_float_acts(s): return np.array(list(map(float, re.findall('(-?\d+.\d+)', s))))

def get_action_dist(act, optima:np.ndarray):
    x = __get_float_acts(act)
    return np.linalg.norm(x - optima)
