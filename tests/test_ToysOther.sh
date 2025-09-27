#!/bin/zsh

ALGORITHMS=("spsa-const" "spsa-ordin" "spsa-multi" "spsa1s-ordin" "4point-ordin" "4point-multi")
ENVIRONMENTS=("Gaussian" "Exponential" "Gamma" "Lognormal" "Pareto")
METHODS=("iterative" "vanilla" "batch")

for env in "${ENVIRONMENTS[@]}"; do
    for alg in "${ALGORITHMS[@]}"; do
        for method in "${METHODS[@]}"; do
            python ./toy_test.py --algorithm "$alg" --alg_config ./_configs/alg_config_random1.json --environment "$env" --env_config ./_configs/env_config_default0.json --len_iteration 102400 --num_replication 300 --folder ./results/$method/ --est_var_method "$method" 
        echo "Finish $alg $env via $method"
        done
    done
done

