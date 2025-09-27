import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

from .base import algorithms



xx = np.linspace(-3, 3, 100)
yy = norm.pdf(xx)

def z_hist_plot(ax, z_scores, *, grid=True, truncate=True, **kwargs):
    if truncate:
        z_show = z_scores[abs(z_scores-z_scores.mean()) <= 3*z_scores.std()]
    else:
        z_show = z_scores
    ax.hist(z_show, density=True, **kwargs)
    ax.plot(xx, yy, color="orange", linestyle="-.", label="stand. normal")
    ax.grid(grid)



def get_zscores(
        read_dir,
        opt_val,
        *,
        batch_size=50,
        rate=0.05,
        hyperparmeter=3,
        rate_mu=0.666,
        **kwargs
):
    z_scores = dict()  # {alg: dict() for alg in algorithms}

    for alg in algorithms:
        bbo, ssi = alg.split("-")
        files_dir = f"{read_dir}/{bbo}_{ssi}/"
        mu_ests = pd.read_csv(f"{files_dir}/mu_estimates.csv", header=0, index_col=0)
        var_ests = pd.read_csv(f"{files_dir}/var_estimates.csv", header=0, index_col=0)
        # mu_show = mu_ests[abs(mu_ests-mu_ests.mean()) <= mu_ests.std()]
        # ...
        if ssi == "const":
            z_scores[alg] = (mu_ests - opt_val) / np.sqrt(rate / (2-rate) * var_ests / (2*batch_size))
        else:
            n_iter = np.array(mu_ests.columns, dtype=int)
            if n_iter[0] == 0: n_iter[0] = 1
            if ssi == "ordin":
                z_scores[alg] = (mu_ests - opt_val) / np.sqrt(var_ests / (2*batch_size) / n_iter)
            elif ssi == "multi":
                z_scores[alg] = (mu_ests - opt_val) / np.sqrt(var_ests / (2*batch_size) / n_iter**rate_mu)
        
        if bbo == "4point":
            z_scores[alg] = z_scores[alg] * (hyperparmeter**2 -1) / (hyperparmeter**2 +1)
    
    return z_scores



def get_cover_probs(df, p=0.95):
    return (df.abs() <= norm.ppf((1+p)/2)).mean()
