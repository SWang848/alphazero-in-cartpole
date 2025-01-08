#!bin/bash

# mcts script
for c_init in 2 3 4; do
    for num_simulations in 120 150; do
        sbatch CC_script.sh $c_init $num_simulations 
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


