import os
import pandas as pd
import time
from multiprocessing import Pool

algorithms = [
    "spsa-const",
    "spsa-ordin",
    "spsa-multi",
    "spsa1s-ordin",
    "4point-ordin",
    "4point-multi"
]

# var_methods = ["vanilla"]
var_methods = ["batch", "iterative", "vanilla"]

# envs = ["Bernoulli", "Gaussian", "Exponential", "Gamma", "Lognormal", "Pareto"]
envs = ["PortfolioNormal"]

file_names = {
    "distances": "distances.csv", 
    "estimates": "mu_estimates.csv", 
    "var_ests": "var_estimates.csv",
    "gaps": "optimality_gaps.csv",
    # "run_times": "durations.csv"
}

read_dir = "./results/"

indexes = [0, 1] + [100*2**k for k in range(11)]


def xform(file_path, output_dir):
    begin_time = time.time()

    df = pd.read_csv(file_path, header=0, index_col=0)
    df.columns = list(range(df.shape[1]))

    os.makedirs(output_dir, exist_ok=True)

    df_ = pd.DataFrame({
        "mean": df.mean(),
        "std": df.std()
    })
    df_.to_csv(f"{output_dir}/mean_std.csv", index=False)

    df.iloc[:,indexes].to_csv(f"{output_dir}/hist_data.csv")
    
    end_time = time.time()
    print(
        f"Finish transform",
        f"\t{output_dir}",
        f"\trunning time: {(end_time - begin_time)/60:.8f} minute(s).",
        "="*50, sep="\n"
    )


def move(file_path, output_path):
    df = pd.read_csv(file_path, header=None)
    df.to_csv(output_path, index=None)

read_dir = "./results/portfolio_positive/"
# read_dir = "./results/portfolionormal/"
write_dir = "./_positive_portfolio/"

# read_dir = "./results/portfolio0/"
# write_dir = "./_zero_portfolio/"

# read_dir = "./results/"
# write_dir = "./_toy_data/"



if __name__ == "__main__":
    args = []
    time_io = []

    for var_method in var_methods:
        for alg in algorithms:
            bbo, ssi = alg.split("-")
            for env in envs:
                for key, f_name in file_names.items():
                    file_path = f"{read_dir}/{var_method}/{env}/{bbo}_{ssi}/{f_name}"
                    # output_dir = f"{write_dir}/{var_method}/{env}/{bbo}_{ssi}/{key}/"
                    output_dir = f"{write_dir}/{var_method}/{bbo}_{ssi}/{key}/"
                    args.append((file_path, output_dir))
                time_in = f"{read_dir}/{var_method}/{env}/{bbo}_{ssi}/durations.csv"
                # time_out = f"{write_dir}/{var_method}/{env}/{bbo}_{ssi}/run_times.csv"
                time_out = f"{write_dir}/{var_method}/{bbo}_{ssi}/run_times.csv"
                time_io.append((time_in, time_out))


    with Pool(processes=5) as pool:
        pool.starmap(xform, args)
        # pool.starmap(print, args)
    

    with Pool(processes=5) as pool:
        pool.starmap(move, time_io)

    