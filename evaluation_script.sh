#!/bin/bash

python main.py --num_envs_per_worker 1 --num_simulations 70 --opr "test" --num_rollout_worker 1 --num_cpus_per_worker 16 \
                --num_gpus_per_worker 1 --num_test_episode 1 --num_target_blocks 15 --c_init 3.5 --model_path /home/swang848/efficientalphazero/results/Swap-v0_07012025_2327_59/model_latest.pt