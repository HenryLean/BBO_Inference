import numpy as np
import random
import time
import argparse
import json

from functools import partial
from multiprocessing import Pool, Manager
from __src import *


estimates = ["actions", "estimates", "variances"]
algs = {
    "4-point": FourPoint,
    "spsa": SPSA,
    "kwsa": KWSA,
    "kwsa1": KWSA_1side
}
infs = {
    "constant": ConstInfer,
    "4-point": FourPointInfer,
    "ordinary": OrdinaryInfer
}
envs = {
    "binary": BinaryEnv,
    "gaussian": GaussianEnv,
    "exponent": ExponentialEnv
}
mean_fns = {
    "quadratic": quadratic_fn,
    "logistic": logistic_fn
}
noise_fns = {
    "constant": get_const,
    "periodic": periodic_fn,
}
batch_fns = {
    "constant": get_const,
    "constant4": get_const4,
}


def run(lock, algName, infName, envName, seed, name, config_dir="./input/test.json"):
    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    print("{}_{}_{}#{} begins at {:s}".format(algName, infName, envName, name, time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(start_time))))

    args = Option(algName, infName, envName, config_dir).parse_args()
    algArgs = {"act": args.init_act, "bounds": args.bounds, "hypara": args.alg_hypara}
    infArgs = (args.init_est, args.init_var, args.inf_hypara)
    env_fns = {
        "meanFn": partial(mean_fns[args.env_mean], **args.env_mean_kwargs)
    }
    if envName == "gaussian":
        env_fns["noiseFn"] = partial(noise_fns[args.env_noise], **args.env_noise_kwargs)

    kwfns = {
        "perturb_fn": partial(get_harmony, **args.pert_kwargs),
        "step_fn": partial(get_harmony, **args.step_kwargs),
        "batch_fn": partial(batch_fns[args.batch_type], **args.batch_kwargs)
    }

    bboInfer = BboInfer(algs[algName], infs[infName], *infArgs, **algArgs)
    env = envs[envName](**env_fns) # some functions
    indices = [0] + [(k+1)*10**i-1 for i in range(int(np.log10(args.horizon))) for k in range(1, 10)]
    bboInfer.run(env, N=args.horizon, indices=indices, **kwfns)
    with lock:
        for key in estimates:
            bboInfer.write(key, output_dir=output_dir+"{}_{}_{}_".format(algName, infName, envName))

    end_time = time.time()
    print(
        "{}_{}_{}#{} begins at {:s}".format(algName, infName, envName, name, time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(start_time))), 
        "ends at {:s}".format(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(end_time))),
        # "costs {:.2f} mins in total.\n".format((end_time - start_time)/60)
        "costs {:.2f} ms in total.\n".format(1000*(end_time - start_time)), sep="\n"
    )
    
    
def multirun(r, alg_name, inf_name, env_name, max_p:int=10):
    manager = Manager()
    lock = manager.RLock()
    locks = [lock] * r
    algNames = [alg_name] * r
    infNames = [inf_name] * r
    envNames = [env_name] * r
    seeds = [111*i for i in range(1,1+r)]
    names = list(range(1,1+r))
    zipped_list = list(zip(locks, algNames, infNames, envNames, seeds, names))

    num = min(r, max_p)
    pool = Pool(processes=num)
    pool.starmap(partial(run, config_dir=config_dir), zipped_list)
    pool.close()
    pool.join()


class Option(object):
    def __init__(self, alg_name, inf_name, env_name, config_dir="./input/config.json"):
        self.alg_name = alg_name
        self.inf_name = inf_name 
        self.env_name = env_name
        self.config_dir = config_dir

    def parse_args(self, **kwargs):
        with open(self.config_dir, "r") as f:
            config = json.load(f)
        parser = argparse.ArgumentParser()
        # parsing algorithmic arguments
        parser.add_argument("--init_act", type=np.ndarray, default=config["algArgs"]["act"])
        parser.add_argument("--bounds", type=dict, default=config["algArgs"]["bounds"])
        parser.add_argument("--alg_hypara", type=float, default=config["algArgs"]["hypara"][self.alg_name])
        # parsing inferencial arguments
        parser.add_argument("--init_est", type=float, default=config["infArgs"]["est"])
        parser.add_argument("--init_var", type=float, default=config["infArgs"]["var"])
        parser.add_argument("--inf_hypara", type=float, default=config["infArgs"]["hypara"][self.inf_name])
        parser.add_argument("--horizon", type=int, default=int(config["horizon"]))

        parser.add_argument("--pert_kwargs", type=dict, default=config["pert_kwargs"])
        parser.add_argument("--step_kwargs", type=dict, default=config["step_kwargs"])
        parser.add_argument("--batch_type", type=str, default=config["batch"]["batch_type"])
        # parsing environmental arguments
        parser.add_argument("--env_mean", type=str, default="quadratic")  # Manual setting
        args = parser.parse_args()
        batch_type = args.batch_type
        mean_type = args.env_mean
        parser.add_argument("--batch_kwargs", type=dict, default=config["batch"]["batch_kwargs"][batch_type])
        parser.add_argument("--env_mean_kwargs", type=dict, default=config["env"][self.env_name][mean_type]["mean_kwargs"])
        if self.env_name == "gaussian":
            parser.add_argument("--env_noise", type=str, default="periodic")  # Manual setting
            noise_type = parser.parse_args().env_noise
            parser.add_argument("--env_noise_kwargs", type=dict, 
            default=config["env"][self.env_name][noise_type]["noise_kwargs"])
        args = parser.parse_args(**kwargs)
        return args

config_dir = "./input/test.json"
output_dir = "./output/csv/"


if __name__ == "__main__":
    begin_time = time.time()
    r = 1000
    for alg_name, inf_name, env_name in [("4-point", "4-point", "gaussian"), ("spsa", "constant", "gaussian")]:
    # output_dir+"{}_{}_{}_".format(alg_name, inf_name, env_name)
        multirun(r, alg_name, inf_name, env_name)
    end_time = time.time()
    print("total time: {:.2f} ms.".format((end_time - begin_time)*1000))
    # print("total time: {:.2f} minutes.".format((end_time - begin_time)/60))
