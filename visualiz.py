import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
from __utils.functions import *
from __utils.plots import *

horizon = 1e4
indices = [0] + [(k+1)*10**i-1 for i in range(int(np.log10(horizon))) for k in range(1, 10)]
names = ["actions", "estimates", "variances"]
mean_fns = {
    "bin1": quadrFn,
    "prt2": quadrFn,
    "exp3": quadrFn,
    "gmm4": quadrFn,
    "lnm5": quadrFn,
    "quadratic": quadratic_fn,
    "logistic": logistic_fn,
    "perm": fnPerm,
    "permBin": lambda args, **kwargs: sigmoid(fnPerm(args, **kwargs)),
    "trid": fnTrid,
    "tridBin": lambda args, **kwargs: sigmoid(fnTrid(args, **kwargs))
}
kwargs = {
    "bin1": {"A": [[-0.02125]], "b": [0.01825], "intercept": 0.0105},
    "prt2": {"A": [[-0.02125]], "b": [0.01825], "intercept": 0.0105},
    "exp3": {"A": [[-0.02125]], "b": [0.01825], "intercept": 0.0105},
    "gmm4": {"A": [[-0.02125]], "b": [0.01825], "intercept": 0.0105},
    "lnm5": {"A": [[-0.02125]], "b": [0.01825], "intercept": 0.0105},
    "quadratic": {"L": [[1, 0], [-0.2, 1]], "b": [-1, 0.5], "intercept": 1},
    "logistic": {"L": [[1, 0], [-0.2, 1]], "b": [-1, 0.5], "intercept": -1},
    "perm": {"intercept": 1.0},
    "permBin": {"scale":0.1, "intercept": -1.0},
    "trid": {"scale": 0.01, "intercept": 10},
    "tridBin": {"scale": 0.001, "intercept":-1}
}
hypara = {
    "constant": 0.05,
    "multi_TS": 0.666, 
    "4-point": 3,
    "ordinary": None
}
linestyles = {
    "4-point": "-",
    "multi_TS": "--",
    "ordinary": "-",
    "constant": "-."
}
markers = {
    "spsa": ".",
    "4-point": "d",
    "spsa1s": "*"
}


def get_optimiser(L, b, *args, **kwargs):
    L = np.array(L, float)
    b = np.array(b, float)
    return np.linalg.inv(L.T @ L) @ b

def dist_plot(df, optimiser:np.ndarray, get_dist=False, *args, **kwargs):
    df_dist = df.apply(lambda row: row.apply(partial(get_action_dist, optimiser=optimiser)))
    cvg_plot(df_dist, true=0, *args, **kwargs)
    final_dists = df_dist.iloc[:,-1]
    print("final dist = {:.4f} ({:.4f})".format(np.mean(final_dists), np.std(final_dists)))
    return df_dist if get_dist else None

def get_mean_std(df, *args, **kwargs):
    df = pd.concat([df.mean(), df.std()],axis=1,ignore_index=True)
    return df.apply(lambda row: "{:.4f}({:.4f})".format(row[0], row[1]), axis=1)

def get_tStats(dfm, dfv, optimum, inf, hypara, histIndex, batch=40, *args, **kwds):
    histIndex = np.array(histIndex, int)
    if inf == "4-point":
        sclr = np.sqrt((histIndex+1)*batch)*(hypara**2-1)/(hypara**2+1)
        t_stats = (dfm[histIndex] - optimum)/np.sqrt(dfv[histIndex])*sclr
    elif inf == "constant":
        t_stats = (dfm[histIndex] - optimum)/np.sqrt(dfv[histIndex]*hypara/(2-hypara)/batch)
    elif inf == "multi_TS":
        t_stats = (dfm[histIndex] - optimum)/np.sqrt(dfv[histIndex]/(1+histIndex)**hypara/batch)
    else:
        t_stats = (dfm[histIndex] - optimum)/np.sqrt(dfv[histIndex]/(1+histIndex)/batch*2)
    return t_stats

def get_coverage(t_stats, z=1.96): return np.mean(t_stats.apply(abs) < z)

def hist_subplots(ax, stats, **kwargs):
    a, b = min(stats), max(stats)
    xx = np.linspace((a-b)/2, (b-a)/2, 100)
    yy = norm.pdf(xx)
    ax.hist(stats, density=True, label="histogram", **kwargs)
    ax.plot(xx, yy, linestyle="-.", label="std. normal")
    ax.axvline(0, color="red", linestyle=":")
    ax.grid(True)
    ax.legend()

def hist_cvg_plots(nrows, ncols, t_stats, r=1, c=1, d=1, bins=25, plot_dir="", *args, **kwargs):
    _, ax = plt.subplots(nrows, ncols, *args, **kwargs)
    for i in range(nrows):
        for j in range(ncols):
            index = (c*j+1)*10**(r*i+d)-1
            hist_subplots(ax[i, j], t_stats[index], bins=bins)
            ax[i, j].set_title("{:d}e{:d} iterations".format(c*j+1, r*i+d))
    if plot_dir:
        try:
            plt.savefig(plot_dir)
        except: pass
    else:
        plt.show()
    plt.close()

def coverage_plot(df, linestyles, markers, t=0, title="", plot_dir="", *args, **kwargs): 
    indexes = df.index[t:]
    plt.figure(*args, **kwargs)
    for col in df.columns:
        alg, inf, env = col.split("*")
        plt.plot(np.log10(indexes+1), df.loc[indexes, col], linestyle=linestyles[inf], marker=markers[alg], label="{:s}+{:s}".format(alg, inf))
    plt.axhline(.95, linestyle=":", color="red", label="confidence level")
    plt.title(title)
    plt.xlabel(r"$log_{10}$(#iterations)")
    plt.grid()
    plt.legend()
    if plot_dir:
        try:
            plt.savefig(plot_dir)
        except: pass
    else:
        plt.show()
    plt.close()


def run(alg, inf, env, meanType, optimiser, csv_dir, output_dir, batch=40, **kwds):
    start_time = time.time()

    key = "{:s}_{:s}_{:s}_".format(alg, inf, env)
    read_dir = csv_dir + key
    fig_dir = output_dir + "figures/" + key

    print("begin...{:s} {:s} {:s}".format(alg, inf, time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(start_time))))
    optimum = mean_fns[meanType](optimiser, **kwds)
    print("optimal objective: {:.4f}".format(optimum))

    df = pd.read_csv(read_dir+"actions.csv", header=None)
    df.columns = indices
    df_dist = dist_plot(df, optimiser, tail=0, logIndex=True, get_dist=True, plot_dir=fig_dir+"dist.jpg", dpi=900)

    dfm = pd.read_csv(read_dir+"estimates.csv", header=None)
    dfm.columns = indices
    cvg_plot(dfm, optimum, tail=0, logIndex=True, ciType="empir", plot_dir=fig_dir+"cvg.jpg", dpi=900)

    dfv = pd.read_csv(read_dir+"variances.csv", header=None)
    dfv.columns = indices
    t_stats = get_tStats(dfm, dfv, optimum, inf, hypara[inf], indices, batch)
    cit_plot(t_stats.iloc[:,-1], dpi=300, plot_dir=fig_dir+"_hist.jpg".format(meanType, key))
    hist_cvg_plots(2, 5, t_stats, r=2, c=2, d=2, dpi=300, figsize=(20,12), plot_dir=fig_dir+"_histCVG.jpg")

    end_time = time.time()
    print("{}_{}_{}: {:.4f} ms.\n".format(alg, inf, env, (end_time - start_time)*1000))
    return get_mean_std(df_dist), t_stats.apply(partial(get_coverage, z=1.96))


if __name__ == "__main__":
    start_time = time.time()

    # meanType = "quadratic"
    # optimiser = get_optimiser(**kwargs[meanType])

    # meanType, env_name= "bin1", "binary"
    # meanType, env_name= "prt2", "pareto"
    # meanType, env_name= "exp3", "exponent"
    # meanType, env_name= "gmm4", "gamma"
    meanType, env_name= "lnm5", "lognorm"
    optimiser = [0.01825/2/0.02125]

    # csv_path, output_path = "result/%s/csv/"%env_name, "result/%s/outputs/"%env_name
    csv_path, output_path = "output/%s/csv/"%meanType, "output/%s/result/"%meanType

    testList = [  # "pareto", "lognorm", "exponent", "gamma"
        ("spsa", "ordinary", env_name),
        ("spsa", "multi_TS", env_name),
        ("spsa", "constant", env_name),
        ("spsa1s", "ordinary", env_name),
        ("4-point", "4-point", env_name),
        ("4-point", "multi_TS", env_name)
    ]
    opt_list = ["spsa","spsa1s","4-point"]
    fail_list = []
    df_tab = pd.DataFrame()
    df_cvr = pd.DataFrame()
    for alg, inf, env in testList:
        dist, cvr = run(alg, inf, env, meanType, optimiser, csv_path, output_path, batch=100, **kwargs[meanType])
        if alg in opt_list:
            df_tab.insert(0, alg, dist)
            opt_list.remove(alg)
        df_cvr.insert(0, "{}*{}*{}".format(alg, inf, env), cvr)
        # try:
        #     dist, cvr = run(alg, inf, env, meanType, optimiser, csv_path, output_path, batch=100, **kwargs[meanType])
        #     if alg in opt_list:
        #         df_tab.insert(0, alg, dist)
        #         opt_list.remove(alg)
        #     df_cvr.insert(0, "{}*{}*{}".format(alg, inf, env), cvr)
        # except:
        #     fail_list.append("{} {} {}".format(alg, inf, env))
    coverage_plot(df_cvr, linestyles, markers, t=9, dpi=900, figsize=(10,6), plot_dir=output_path+"figures/{:s}_coverage.jpg".format(meanType))
    df_tab.to_csv(output_path+"tabulars/{:s}_dist.csv".format(meanType))
    df_cvr.to_csv(output_path+"tabulars/{:s}_coverage.csv".format(meanType))

    end_time = time.time()
    print("{:.4f} seconds in total\n".format((end_time - start_time)))

