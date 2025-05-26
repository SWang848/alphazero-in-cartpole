from argparse import ArgumentParser
import os
import ray
import torch
import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import time

from config.place import Config
from core.storage import add_logs
from core.workers import TestWorker

import logging

logging.getLogger("ray").setLevel(logging.ERROR)


def policy_track(args, config, model):
    start_time = time.time()
    test_workers = [
        TestWorker.options(
            num_cpus=args.num_cpus_per_worker, num_gpus=args.num_gpus_per_worker
        ).remote(config, args.device_workers, args.amp)
        for _ in range(args.num_rollout_workers)
    ]
    num_episodes_per_worker = int(args.num_test_episodes / args.num_rollout_workers)
    workers = [
        test_worker.run.remote(model.get_weights(), num_episodes_per_worker)
        for test_worker in test_workers
    ]

    ray.wait(workers)

    evaulation_stats_all = {}  # Accumulate evaluation stats
    for i, test_worker in enumerate(test_workers):
        test_stats, evaulation_stats_all, best_found = ray.get(
            test_worker.get_stats.remote()
        )
        add_logs(test_stats, evaulation_stats_all)

    stats = {}
    action_trajectory = []
    reward_trajectory = []
    for i in range(len(evaulation_stats_all["action"])):
        stats["action"] = evaulation_stats_all["action"][i]
        stats["reward"] = evaulation_stats_all["reward"][i]
        # stats["info"] = evaulation_stats_all["info"][i]
        stats["mcts_policy"] = evaulation_stats_all["mcts_policy"][i]
        stats["value_target"] = evaulation_stats_all["value_target"][i]
        action_trajectory.append(evaulation_stats_all["action"][i])
        # print(f"step: {i}\n")
        # print(stats)
        reward_trajectory.append(evaulation_stats_all["reward"][i])
    # np.savez(os.path.join(log_dir, "best_found.npz"), hpwl=best_found["hpwl"], place_infos=best_found["state"].place_infos)
    wall_clock_time = time.time() - start_time
    print(f"action:{action_trajectory}, reward:{reward_trajectory}")
    print(
        f"the best hpwl is: {best_found['hpwl']}, the best reward is: {best_found['reward']}"
    )
    print(f"Wall clock time: {wall_clock_time:.2f} seconds")

    for worker in test_workers:
        ray.kill(worker)

    return best_found["hpwl"], wall_clock_time


if __name__ == "__main__":
    parser = ArgumentParser("MCTS Place, GO")
    parser.add_argument("--env", type=str, default="Classic-v0")
    # parser.add_argument("--env", type=str, default="Classic-v0")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--num_rollout_workers", default=1, type=int)
    parser.add_argument("--num_cpus_per_worker", default=16, type=float)
    parser.add_argument("--num_gpus_per_worker", default=1, type=float)
    parser.add_argument("--num_test_episodes", default=1, type=float)
    parser.add_argument("--model_path", default="/home/swang848/efficientalphazero/results_RL4RS/c15b_mcts_RL4RS_seed_7119/model_best.pt")
    # parser.add_argument(
    #     "--model_dir_pool",
    #     nargs="+", 
    #     default=["/home/swang848/efficientalphazero/results_RL4RS/c15b_gumbel_RL4RS_seed_6757/model_best.pt"],
    #     help="List of model directories to evaluate"
    # )
    parser.add_argument("--device_workers", default="cuda", type=str)
    parser.add_argument("--device_trainer", default="cuda", type=str)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_target_blocks", default=15, type=int)
    parser.add_argument("--place_order", default="connections_", type=str)
    parser.add_argument("--c_init", default=8.0, type=float)
    parser.add_argument("--num_simulations", default=150, type=int)
    parser.add_argument("--num_envs_per_worker", default=1, type=int)
    parser.add_argument("--value_support_min", default=-1, type=int)
    parser.add_argument("--value_support_max", default=0, type=int)
    parser.add_argument("--value_support_delta", default=0.1, type=float)
    parser.add_argument("--non_fixed_init", action="store_true")
    parser.add_argument("--forced_exploration", action="store_true")
    parser.add_argument("--k", default=2.0, type=float)
    parser.add_argument("--percentage", default=0.3, type=float)

    args = parser.parse_args()

    sub_dir = datetime.now().strftime("%d%m%Y_%H%M")
    sub_dir = f"{args.env}_{sub_dir}_{random.randint(10, 99)}"
    # if program is run on CC, save logs to the local disk.
    if args.debug:
        sub_dir = f"debug/{sub_dir}"
    if os.path.isabs(args.results_dir):
        log_dir = os.path.join(args.results_dir, sub_dir)
    else:
        log_dir = os.path.join(os.getcwd(), args.results_dir, sub_dir)

    os.makedirs(log_dir, exist_ok=True)
    config = Config(
        log_dir=log_dir,
        value_support_max=args.value_support_max,
        value_support_min=args.value_support_min,
    )  # Apply set BaseConfig arguments

    for arg, arg_val in vars(args).items():
        if hasattr(config, arg):
            setattr(config, arg, arg_val)
            print(f'Overwriting "{arg}" config entry with {arg_val}')
        else:
            setattr(config, arg, arg_val)
            print(f'Adding "{arg}" config entry with {arg_val}')

    setattr(
        config,
        "replay_buffer_size",
        args.num_rollout_workers
        * config.min_num_episodes_per_worker
        * config.num_target_blocks
        * 2,
    )
    print(
        f'Overwriting "replay_buffer_size" config entry with {args.num_rollout_workers * config.min_num_episodes_per_worker * config.num_target_blocks * 4}'
    )

    print(args)

    model = config.init_model(args.device_trainer, args.amp)  # Create (and load) model
    model.load_state_dict(torch.load(args.model_path))

    ray.init(log_to_driver=False)
    
    # Run experiments with different simulation budgets
    simulation_budgets = range(1, 101, 5)  # From 1 to 100 with step size 5
    results = {
        'simulation_budgets': [],
        'best_hpwl': [],
        'wall_clock_time': []
    }
    
    for num_simulations in simulation_budgets:
        print(f"\nRunning with simulation budget: {num_simulations}")
        setattr(config, "num_simulations", num_simulations)
        best_hpwl, wall_clock_time = policy_track(args, config, model)
        
        results['simulation_budgets'].append(num_simulations)
        results['best_hpwl'].append(best_hpwl)
        results['wall_clock_time'].append(wall_clock_time)
    
    # Save results to file
    results_file = os.path.join(log_dir, "simulation_budget_results.npz")
    np.savez(
        results_file,
        simulation_budgets=results['simulation_budgets'],
        best_hpwl=results['best_hpwl'],
        wall_clock_time=results['wall_clock_time']
    )
    print(f"\nResults saved to {results_file}")
    
    # Print summary
    print("\nSummary of results:")
    for i in range(len(results['simulation_budgets'])):
        print(f"Simulations: {results['simulation_budgets'][i]}, "
              f"Best HPWL: {results['best_hpwl'][i]:.2f}, "
              f"Time: {results['wall_clock_time'][i]:.2f}s")
   