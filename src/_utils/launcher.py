import argparse
import json
import numpy as np
from functools import partial
from types import SimpleNamespace

from ..algorithms import *
from ..environments import *
from .functions import QuadraticFn, SigmoidQuadrFn, SineFn

from gradopt.gradient_descent import poly_lr


bbo_algorithms = {
    "spsa": bbo.BaseSPSA, 
    "spsa1s": bbo.BaseSPSA1s, 
    "4point": bbo.Base4Point
}

ssi_alogrithms = {
    "const": ssi.ConstSSI, 
    "ordin": ssi.OrdinSSI, 
    "multi": ssi.MultiSSI
}

algorithm_options = [
    "spsa-const",
    "spsa-ordin",
    "spsa-multi",
    "spsa1s-ordin",
    "4point-ordin",
    "4point-multi"
]

environments = {
    "Bernoulli": BinaryEnv,
    "Gaussian": GaussianEnv,
    "Exponential": ExponentialEnv,
    "Gamma": GammaEnv,
    "Lognormal": LogNormEnv,
    "Pareto": ParetoEnv,
    "PortfolioNormal": PortfolioNormalEnv,
    "PortfolioJump": PortfolioJumpEnv,
}

functions = {
    "quadr": QuadraticFn,
    "sigmoid": SigmoidQuadrFn,
    "sine": SineFn
}

env_portfolio = ["PortfolioNormal", "PortfolioJump"]


def opt_toy(mean, mean_func, **kwargs):
    A = np.array(mean["A"], dtype=float)
    b = np.array(mean["b"], dtype=float)
    theta_opt = np.linalg.inv(A) @ b
    
    c = float(mean["c"])
    if mean_func == "quadr":
        return theta_opt, c - 0.5 * b @ theta_opt
    
    if mean_func == "sigmoid":
        return theta_opt, 0.99 / (1 + np.exp(0.5 * b @ theta_opt - c)) + 0.005

    return theta_opt, 0



class Option:
    def __call__(self, *args, **kwds):
        parser = argparse.ArgumentParser()
        parser.add_argument("--algorithm", choices=algorithm_options, default="spsa-const", type=str, help=f"Zeroth order optimization with simultaneous statistical inference for optimal performance algorithm.\n\tOptions={algorithm_options}")
        parser.add_argument("--environment", choices=list(environments.keys()), default="PortfolioNormal", type=str, help=f"The risky environment.\n\tOptions={list(environments.keys())}")
        parser.add_argument("--env_config", default=None, type=str)
        parser.add_argument("--alg_config", default=None, type=str)
        parser.add_argument("--batch_size", default=50, type=int)
        parser.add_argument("--len_iteration", default=10000, type=int)
        parser.add_argument("--num_replication", default=3, type=int)
        parser.add_argument("--folder", default="./_cache/", type=str)
        parser.add_argument("--rec_indexes", default=None, type=str)
        parser.add_argument("--est_var_method", default="iterative", choices=["iterative", "batch", "vanilla"], type=str)
        return parser.parse_args()


    def _parse_args(self, verbose=False):
        args = self.__call__()

        bbo_name, ssi_name = args.algorithm.split("-")
        env_name = args.environment

        if args.alg_config is None:
            alg_config = "./_configs/alg_config_default2.json" if env_name in env_portfolio else "./_configs/alg_config_default1.json"
        else: 
            alg_config = args.alg_config

        if args.env_config is None:
            env_config = "./_configs/env_config_default2.json" if env_name in env_portfolio else "./_configs/env_config_default1.json"
        else:
            env_config = args.env_config
        
        with open(alg_config, "r") as f:
            alg_args = json.load(f)

        with open(env_config, "r") as f:
            env_args = json.load(f)


        if verbose:
            num_collect = args.len_iteration if args.rec_indexes is None else len(args.rec_indexes)
            print(
                "="*100, 
                " "*40 + "PARSING  ARGUMENTS", 
                " "*40 + "Detailed values:", 
                "="*100,
                f"Blackbox optimization algorithm = ``{bbo_name}``", 
                f"Simutaneous statistical inference = ``{ssi_name}``", 
                f"Environment class = ``{env_name}``", 
                f"Algorithm arguments = {alg_args}", 
                f"Environment arguments = {env_args}",
                f"Each iteration take a sample of size {args.batch_size}",
                f"Run each path with {args.len_iteration} iteration(s)",
                f"Replicate {args.num_replication} homogeneous path(s)",
                f"Collect data of size {num_collect} in ``{args.folder}``",
                f"Variance estimate method: {args.est_var_method}", 
                "="*100, sep = "\n"
            )
        if args.rec_indexes is not None:
            with open(args.rec_indexes, "r") as f:
                args.rec_indexes = json.load(f)

        output = {
            "bbo_name": bbo_name,
            "ssi_name": ssi_name,
            "env_name": env_name,
            "alg_args": alg_args,
            "env_args": env_args,
            "batch_size": args.batch_size,
            "len_iteration": args.len_iteration,
            "num_replication": args.num_replication,
            "folder": args.folder,
            "rec_indexes": args.rec_indexes,
            "est_var_method": args.est_var_method
        }
        
        return SimpleNamespace(**output)



def launch(
    bbo_name, 
    ssi_name, 
    env_name, 
    alg_args, 
    env_args,
    *, 
    seed=None,
    N=10000,
    batch_size=50,
    rec_indexes=None,
    optimizer=0,
    opt_val=0,
    est_var_method="iterative",
    **kwargs
):
    init_act, init_mu, init_var = alg_args["init"]
    bounds = alg_args["bounds"]
    hyperparam = alg_args["hyperparam"]
    rate = alg_args["const_rate"]
    rate_mu, rate_var = alg_args["multi_rates"]
    ssi_kwargs = {"rate":rate, "rate_mu":rate_mu, "rate_var":rate_var}
    perturb_args = alg_args["perturb_args"]
    step_args = alg_args["step_args"]
    if isinstance(init_act, str):
        init_act = None

    if env_name in env_portfolio:
        env_mean = env_args["mean"]
        env_cov = env_args["cov"]
        target = env_args["target"]
        jump_args = env_args["jump_args"]
        env = environments[env_name](env_mean, env_cov, target=target, **jump_args)
    else:
        env_mean = functions[env_args["mean_func"]](**env_args["mean"])
        env_noise = functions[env_args["noise_func"]](**env_args["noise"])
        env = environments[env_name](env_mean, noise=env_noise)

    np.random.seed(seed)
    infer = ssi_alogrithms[ssi_name](init_mu, init_var, **ssi_kwargs)
    bbossi_launcher = bbo_algorithms[bbo_name](init_act, infer, env, bounds, hyperparam, optimizer=optimizer, opt_val=opt_val, **kwargs)

    perturb_fn = partial(poly_lr, **perturb_args)
    step_fn = partial(poly_lr, **step_args)
    batch_fn = lambda k: batch_size
    bbossi_launcher.run(perturb_fn, step_fn, batch_fn, N, indexes=rec_indexes, optimizer=optimizer, opt_val=opt_val, est_var_method=est_var_method)

    return bbossi_launcher.memory
