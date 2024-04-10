import csv
import time
import numpy as np
import pandas as pd
from __utils.functions import *
from __utils.plots import get_action_dist, cvg_plot, cit_plot
from functools import partial

mean_fns = {
    "quadr": quadratic_fn,
    "logit": logistic_fn
}
kwds = {"L": [[1, 0], [-.2, 1]], "b": [-1, .5], "intercept": 1.}
optima = np.array([-.9, .32])
indices = [0] + [(k+1)*10**i-1 for i in range(5) for k in range(1, 10)]

def get_target_cols(csv_dir:str, indices:list):
    with open(csv_dir, "r") as f:
        reader = csv.reader(f)
        result = [[row[i] for i in indices] for row in reader]
    return np.array(result, dtype=float)


def dist_plot(df, optima:np.ndarray, *args, **kwargs):
    df_dist = df.apply(lambda row: row.apply(partial(get_action_dist, optima=optima)))
    cvg_plot(df_dist, true=0, *args, **kwargs)
    final_dists = df_dist.iloc[:,-1]
    print("final dist = {:.4f} ({:.4f})".format(np.mean(final_dists), np.std(final_dists)))


if __name__ == "__main__":
    start_time = time.time()
    meanType = "quadr"
    alg, inf, env = "spsa", "constant", "gaussian"
    # "4-point", "4-point", "gaussian"  # "spsa", "constant", "exponent"
    print("begin...{:s} {:s} {:s}".format(alg, inf, time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(start_time))))
    optimal = mean_fns[meanType](optima, **kwds)
    print("optimal objective: {:.4f}".format(optimal))

    read_dir = "output/csv/{:s}_{:s}_{:s}_".format(alg, inf, env)
    output_dir = "output/figs/{:s}_{:s}_{:s}_".format(alg, inf, env)

    df = pd.read_csv(read_dir+"actions.csv", header=None)
    df.columns = indices

    dist_plot(df, optima, tail=0, logIndex=True, plot_dir=output_dir+"dist.jpg", dpi=900)
    current = time.time()
    print("finishes plot dist... {:s}".format(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(current))))

    dfm = pd.read_csv(read_dir+"estimates.csv", header=None)
    dfm.columns = indices
    cvg_plot(dfm, optimal, tail=0, logIndex=True, plot_dir=output_dir+"cvg.jpg", dpi=900)
    current = time.time()
    print("finishes plot cvg... {:s}".format(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(current))))
    estimates = dfm.iloc[:,-1]
    print("final mean estimate = {:.4f} ({:.4f})".format(np.mean(estimates), np.std(estimates)))

    variances = get_target_cols(read_dir+"variances.csv", [-1])[:,0]
    current = time.time()
    print("finishes reading var... {:s}".format(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(current))))
    print("mean variance = {:.4f} ({:.4f})".format(np.mean(variances), np.std(variances)))
    if inf == "constant":
        stats = (estimates - optimal) / np.sqrt(.05*variances/80)
    elif inf == "4-point":
        h = 3
        sclr = (h**2-1)/(h**2+1) * np.sqrt(4e6)
        stats = (estimates - optimal) / np.sqrt(variances) * sclr  # 4-point
    else:
        stats = (estimates - optimal) / np.sqrt(variances/1e5)
    cit_plot(stats, plot_dir=output_dir+"hist.jpg", dpi=900)

    end_time = time.time()
    print("{:.4f} mins in total".format((end_time - start_time)/60))
