import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


algorithms = [
    "spsa-const",
    "spsa-ordin",
    "spsa-multi",
    "spsa1s-ordin",
    "4point-ordin",
    "4point-multi"
]


def plot_mean_std(ax, mean, std, benchmark=0, *, label_b=None, p=0.9, alpha=0.15, grid=True, **kwargs):
    index = mean.index
    ax.plot(mean, **kwargs)
    z = norm.ppf((1+p)/2)
    ax.fill_between(index, mean-z*std, mean+z*std, alpha=alpha)
    ax.axhline(benchmark, linestyle=":", color="red", label=label_b)
    ax.grid(grid)

