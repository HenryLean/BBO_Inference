METHODS=("iterative" "vanilla" "batch")
ALGORITHMS=("spsa-const" "spsa-ordin" "spsa-multi" "spsa1s-ordin" "4point-ordin" "4point-multi")

for alg in "${ALGORITHMS[@]}"; do
    for method in "${METHODS[@]}"; do
        ### 
        python /Users/henrylean/Gits/bbossi/main.py --algorithm "$alg" --alg_config ./_configs/alg_config_default2.json --num_replication 300 --len_iteration 102400 --folder "./results/$method" --est_var_method "$method" 
        ### 
        # python /Users/henrylean/Gits/bbossi/main.py --algorithm "$alg" --alg_config ./_configs/alg_config_default2.json --env_config ./_configs/env_config_portfolio.json --num_replication 300 --len_iteration 102400 --folder "./results/$method" --est_var_method "$method" 
        echo "Finish $alg PortfolioNormal via $method"
    done
done