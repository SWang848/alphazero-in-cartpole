#!/bin/bash

python main.py --num_envs_per_worker 1 --num_simulations 70 --opr "test" --num_rollout_worker 16 --num_cpus_per_worker 16 \
                --num_gpu_per_worker 1 --num_test_episode 1 --num_target_blocks 15 --c_init 2 