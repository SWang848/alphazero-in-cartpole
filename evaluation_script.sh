#!/bin/bash

python main.py --amp --env Swap-v0 --num_envs_per_worker 1 --num_simulations 120 --opr "test" --num_rollout_worker 1 --num_cpus_per_worker 16 \
                --min_num_episodes_per_worker 20 --num_gpus_per_worker 1 --num_test_episode 1 --num_target_blocks 15 --c_init 3 --value_support_min -10 \
                --value_support_max 0 --value_support_delta 1 --model_path /home/swang848/efficientalphazero/results/Swap-v0_24022025_1328_59/model_latest.pt

  
