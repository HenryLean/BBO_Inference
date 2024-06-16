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
indices = [0] + [(k+1)*10**i-1 for i in range(5) for k in range(1, 10)]

def get_target_cols(csv_dir:str, indices:list):
    with open(csv_dir, "r") as f:
        reader = csv.reader(f)
        result = [[row[i] for i in indices] for row in reader]
    return np.array(result, dtype=float)


def dist_plot(df, optimiser:np.ndarray, *args, **kwargs):
    df_dist = df.apply(lambda row: row.apply(partial(get_action_dist, optimiser=optimiser)))
    cvg_plot(df_dist, true=0, *args, **kwargs)
    final_dists = df_dist.iloc[:,-1]
    print("final dist = {:.4f} ({:.4f})".format(np.mean(final_dists), np.std(final_dists)))


def runPlots(alg, inf, env, meanType, optimiser, csv_dir, fig_dir, **kwds):
    start_time = time.time()
    
    read_dir = csv_dir + "{:s}_{:s}_{:s}_".format(alg, inf, env)
    output_dir = fig_dir + "{:s}_{:s}_{:s}_".format(alg, inf, env)
    
    print("begin...{:s} {:s} {:s}".format(alg, inf, time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(start_time))))
    optimal = mean_fns[meanType](optimiser, **kwds)
    print("optimal objective: {:.4f}".format(optimal))

    df = pd.read_csv(read_dir+"actions.csv", header=None)
    df.columns = indices

    dist_plot(df, optimiser, tail=0, logIndex=True, plot_dir=output_dir+"dist.jpg", dpi=900)
    current = time.time()
    print("finishes plot dist... {:s}".format(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(current))))

    dfm = pd.read_csv(read_dir+"estimates.csv", header=None)
    dfm.columns = indices
    cvg_plot(dfm, optimal, tail=0, logIndex=True, ciType="empir", plot_dir=output_dir+"cvg.jpg", dpi=900)
    current = time.time()
    print("finishes plot cvg... {:s}".format(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(current))))
    estimates = dfm.iloc[:,-1]
    print("final mean estimate = {:.4f} ({:.4f})".format(np.mean(estimates), np.std(estimates)))

    variances = get_target_cols(read_dir+"variances.csv", [-1])[:,0]
    current = time.time()
    print("finishes reading var... {:s}".format(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(current))))
    print("mean variance = {:.4f} ({:.4f})".format(np.mean(variances), np.std(variances)))
    if inf == "constant":
        stats = (estimates - optimal) / np.sqrt(.05*variances/1.95/40)
    elif inf == "4-point":
        h = 3
        sclr = (h**2-1)/(h**2+1) * np.sqrt(1e5*40)
        stats = (estimates - optimal) / np.sqrt(variances) * sclr  # 4-point
    elif inf == "multi_TS":
        stats = (estimates - optimal) / np.sqrt(variances/(1e5)**(0.666)/40)
    else:
        stats = (estimates - optimal) / np.sqrt(variances/1e5/20)
    cit_plot(stats, plot_dir=output_dir+"hist.jpg", dpi=900)

    end_time = time.time()
    print("{}_{}_{}: {:.4f} ms.\n".format(alg, inf, env, (end_time - start_time)*1000))


if __name__ == "__main__":
    start_time = time.time()
    
    meanType = "quadr"
    env_name = "pareto"
    kwargs = {"L": [[1, 0], [-0.2, 1]], "b": [-1, 0.5], "intercept": 1} 
    optimiser = np.array([-.9, .32])
    
    csv_dir, fig_dir= "result/%s/csv/"%env_name, "result/%s/figs/"%env_name
    testList = [
        ("spsa", "ordinary", env_name),
        ("spsa", "multi_TS", env_name),
        # ("spsa", "constant", env_name),
        # ("spsa1s", "ordinary", env_name),
        # ("4-point", "4-point", env_name),
        ("4-point", "multi_TS", env_name)
    ]
    fail_list = []
    for alg, inf, env in testList:
        # runPlots(alg, inf, env, meanType, optimiser, csv_dir, fig_dir, **kwargs)

        try:
            runPlots(alg, inf, env, meanType, optimiser, csv_dir, fig_dir, **kwargs)
        except: 
            fail_list.append("{} {} {}".format(alg, inf, env))
    
    end_time = time.time()
    if fail_list:
        print("fails:", fail_list)
    print("{:.4f} seconds in total\n".format((end_time - start_time)))
