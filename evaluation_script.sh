#!/bin/bash

python main.py --num_envs_per_worker 1 --num_simulations 120 --opr "test" --num_rollout_worker 1 --num_cpus_per_worker 16 \
                --num_gpus_per_worker 1 --num_test_episode 1 --num_target_blocks 15 --c_init 2 --model_path /home/swang848/efficientalphazero/results/Swap-v0_10012025_1111_59/model_15.pt