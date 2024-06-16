import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def cvg_plot(df, true, clevel=.9, tail=0, logIndex=True, ciType="both", plot_dir="", *args, **kwargs):
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
    if ciType in ["both", "asymp"]:
        plt.fill_between(indices, mean-std*z, mean+std*z, alpha=.2, label="asymptotic CI")
    if ciType in ["both", "empir"]:
        plt.fill_between(indices, q_l, q_u, linestyle="-.", alpha=.1, label="empirical CI")
    plt.grid(True)
    plt.legend()
    if plot_dir:
        try:
            plt.savefig(plot_dir)
        except: pass
    else:
        plt.show()
    plt.close()


def cit_plot(stats, bins=15, plot_dir="", **kwargs):
    """
    histogram of t-statistics
    """
    a, b = min(stats), max(stats)
    xx = np.linspace((a-b)/2, (b-a)/2, 100)
    yy = norm.pdf(xx)
    plt.figure(**kwargs)
    plt.hist(stats, density=True, bins=bins, label="histogram")
    plt.plot(xx, yy, linestyle="-.", label="std. normal")
    plt.axvline(0, color="red", linestyle=":")
    plt.grid(True)
    plt.legend()
    if plot_dir:
        try:
            plt.savefig(plot_dir)
        except: pass
    else:
        plt.show()
    plt.close()


def __get_float_acts(s): return np.array(list(map(float, re.findall('(-?\d+.\d+)', s))))


def get_action_dist(act, optimiser:np.ndarray):
    x = __get_float_acts(act)
    return np.linalg.norm(x - optimiser)
