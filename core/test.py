from statistics import mean, median
import os

import ray
import yaml

from core.storage import SharedStorage, add_logs
from core.workers import RolloutWorker, TestWorker

import numpy as np
import matplotlib.pyplot as plt
import torch
import random


def test(args, config, model, log_dir, plot_dict=None):
    print("Starting testing...")
    ray.init()
    print("Ray initialized")

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

    test_stats_all = {}  # Accumulate test stats
    evaulation_stats_all = {} # Accumulate evaluation stats
    for i, test_worker in enumerate(test_workers):
        test_stats, evaulation_stats_all = ray.get(test_worker.get_stats.remote())
        add_logs(test_stats, test_stats_all)
        add_logs(test_stats, evaulation_stats_all)
    opt_rew = [-34.59520000000066, -21.637000000000626, 87.22398000000157, -24.536960000001272, -1.7815600000003542, 32.41208000000006, 48.9424199999994, -16.438239999999496, -15.769820000000436, 73.53977999999915, -27.99797999999919, 317.3229000000006, 100.13808000000108, 138.94956000000047, 0.0]
    opt_action = [40, 63, 81, 51, 40, 19, 30, 30, 29, 106, 97, 52, 40, 31, 97]
    print("res stats ", evaulation_stats_all["reward"])
    print("action stats ", evaulation_stats_all["action"])
    print("opt rew is ", opt_rew)
    print("opt action is ", opt_action)    

    stats = {}
    for i in range(len(evaulation_stats_all["action"])):
        stats["action"] = evaulation_stats_all["action"][i]
        stats["reward"] = evaulation_stats_all["reward"][i]
        # stats["info"] = evaulation_stats_all["info"][i]
        stats["mcts_policy"] = evaulation_stats_all["mcts_policy"][i]
        stats["value_target"] = evaulation_stats_all["value_target"][i]
        #print(f"step: {i}\n")
        #print(stats)
    if plot_dict is not None:
        name = "model" + str(random.randint(0, 1000))
        plot_dict[name] = {}
        plot_dict[name]["reward"] = evaulation_stats_all["reward"]
        plot_dict[name]["action"] = evaulation_stats_all["action"]
        
    accum_stats = {}  # Calculate stats
    for k, v in test_stats_all.items():
        accum_stats[f"{k}_mean"] = float(mean(v))
        accum_stats[f"{k}_median"] = float(median(v))
        accum_stats[f"{k}_min"] = float(min(v))
        accum_stats[f"{k}_max"] = float(max(v))
    #print(yaml.dump(accum_stats, allow_unicode=True, default_flow_style=False))

    with open(os.path.join(log_dir, "result.yml"), "w") as yaml_file:  # Write to file
        yaml.dump(accum_stats, yaml_file, default_flow_style=False)

    print("Testing finished!")

    ray.shutdown()

"""
runs test() on all model_latest.pt files in saved_weight_path
args:
    args: arguments from main.py
    config: config object
    log_dir: log directory
    saved_weight_path: path to saved weights
returns:
    None
"""
def test_multiple(args, config, log_dir, saved_weight_path):
    # for evaluation purposes
    config.num_envs_per_worker = 1
    config.num_simulations = 10
    #args.opr = "test"
    args.num_rollout_workers = 1
    args.num_cpus_per_worker = 1
    args.num_gpus_per_worker = 0
    args.num_test_episodes = 1
    config.num_target_blocks = 15
    config.c_init = 1.25
    #args.model_path = (
    #    "/home/truong/Documents/pytorch/alphazero-in-cartpole/saved_weights/Place-v0_04122024_2259_59/model_latest.pt"
    #)
    model_path_list = []
    for root, dirs, files in os.walk(saved_weight_path):
        for file in files:
            if file == 'model_latest.pt':
                model_path_list.append(os.path.join(root, file))
    print(model_path_list)
    plot_dict = {}
    plot_dict["opt"] = {}
    plot_dict["opt"]["reward"] = [-34.59520000000066, -21.637000000000626, 87.22398000000157, -24.536960000001272, -1.7815600000003542, 32.41208000000006, 48.9424199999994, -16.438239999999496, -15.769820000000436, 73.53977999999915, -27.99797999999919, 317.3229000000006, 100.13808000000108, 138.94956000000047, 0.0]
    plot_dict["opt"]["action"] = [40, 63, 81, 51, 40, 19, 30, 30, 29, 106, 97, 52, 40, 31, 97]
    for path in model_path_list:
        model = config.init_model(args.device_trainer, args.amp)
        model.load_state_dict(torch.load(path, map_location=args.device_trainer))
        test(args, config, model, log_dir, plot_dict=plot_dict)
    plot(plot_dict) 
    return

"""
Plot reward for each model
args:
    plot_dict: {
        "model_name": {
            "reward": [reward1, reward2, ...],
            "action": [action1, action2, ...]
        }, ...
    }
"""
def plot(plot_dict):
    plt.figure(figsize=(10, 6))
    for name, values in plot_dict.items():
        plt.plot(values['reward'], label=name)
    # Adding labels and title
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Plot of r arrays for each name')
    plt.legend()

    # Display the plot
    plt.show()
    return