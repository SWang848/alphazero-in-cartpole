#!bin/bash

# mcts script
# for c_init in 3 4; do
#     for lr in 1e-3 1e-4; do
#         for seed in 0 10 100; do
#             sbatch CC_script.sh $c_init $lr $seed
#         done
#     done
# done

# for seed in 12 0 8764 723 529; do
#     sbatch CC_script.sh $seed
# done

for lr in 5e-4 1e-4 5e-5; do
    for c_init in 1.5 2 2.5; do
        for c_scale in 0.1 1; do
            sbatch CC_script.sh $lr $c_init $c_scale
        done
    done
done

# ppo script
# for lr_a in 1e-4; do
#     for lr_c in 1e-4; do
#         for entropy_coeff in 5e-2 5e-3 5e-4; do
#             sbatch CC_script-ppo.sh $entropy_coeff $lr_a $lr_c
#         done
#     done
# done


