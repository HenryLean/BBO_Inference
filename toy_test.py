import csv
import os
import time
from functools import partial
from multiprocessing import Pool, Manager

from src._utils.launcher import Option, launch, opt_toy


s_gap = 111

file_names = {
    "distances": "distances.csv", 
    "estimates": "mu_estimates.csv", 
    "var_ests": "var_estimates.csv",
    "gaps": "optimality_gaps.csv"
}


def launch_write(lock, bbo_name, ssi_name, env_name, alg_args, env_args, *, output_dir="./_cache/", **kwargs):
    begin_time = time.time()

    results = launch(bbo_name, ssi_name, env_name, alg_args, env_args, **kwargs)
    seed = kwargs["seed"]
    
    to_write = {
        "distances": [seed/s_gap] + results.distances,
        "estimates": [seed/s_gap] + results.estimates,
        "var_ests": [seed/s_gap] + results.var_ests,
        "gaps": [seed/s_gap] + results.regret
    }

    with lock:
        file_handles = {}
        writers = {}
        try:
            for key, f_name in file_names.items():
                file_name = f"{output_dir}/{f_name}"
                f = open(file_name, "a", newline="")
                file_handles[key] = f
                writers[key] = csv.writer(f)

            for key in file_names:
                writers[key].writerow(to_write[key])
            
        finally:
            for f in file_handles.values():
                f.close()
    
    duration = time.time() - begin_time

    print(f"\t{bbo_name}-{ssi_name}-{env_name} {int(seed/s_gap)}\t time = {duration:.4f} second(s).")
    return duration


def wrap_seed(fn, seed): return fn(seed=seed)



if __name__ == "__main__":
    begin_time = time.time()

    option = Option()
    args = option._parse_args(verbose=True)

    output_dir = f"{args.folder}/{args.env_name.lower()}/{args.bbo_name}_{args.ssi_name}/"
    os.makedirs(output_dir, exist_ok=True)
    head_line = [i+1 for i in range(args.len_iteration)] if args.rec_indexes is None else args.rec_indexes
    head_line = ["seed", 0] + head_line
    for key, f_name in file_names.items():
        with open(f"{output_dir}/{f_name}", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(head_line)


    # theta_opt, mean_opt = opt_markowitz(**args.env_args)
    theta_opt, mean_opt = opt_toy(**args.env_args)

    # Multiprocess
    manager = Manager()
    lock = manager.RLock()

    launch_mp = partial(launch_write, lock, args.bbo_name, args.ssi_name, args.env_name, args.alg_args, args.env_args, output_dir=output_dir, N=args.len_iteration, rec_indexes=args.rec_indexes, optimizer=theta_opt, opt_val=mean_opt, descent=True, est_var_method=args.est_var_method)


    params = [(launch_mp, s_gap*i) for i in range(args.num_replication)]

    with Pool(processes=5) as pool:
        durations = pool.starmap(wrap_seed, params)

    with open(f"{output_dir}/durations.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(durations)

    print(f"optimizer = {theta_opt}, optimal value = {mean_opt}")
    print("-"*100, "\n")

    end_time = time.time()
    print(f"Total time = {(end_time - begin_time)/60} minute(s).\n")