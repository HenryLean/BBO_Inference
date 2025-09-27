#!/bin/zsh


ALGORITHMS=("spsa-const" "spsa-ordin" "spsa-multi" "spsa1s-ordin" "4point-ordin" "4point-multi")
METHODS=("iterative" "vanilla" "batch")

for alg in "${ALGORITHMS[@]}"; do
    for method in "${METHODS[@]}"; do
        python ./toy_test.py --algorithm "$alg" --alg_config ./_configs/alg_config_default1.json --environment Bernoulli --env_config ./_configs/env_config_default1.json --num_replication 300 --len_iteration 102400 --folder ./results/$method/
        echo "Finish $alg Bernoulli via $method"
    done
done

