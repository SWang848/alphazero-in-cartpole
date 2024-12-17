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


def test(args, config, model, log_dir, file_name, plot_dict=None):
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

    hpwl_list = []
    for item in evaulation_stats_all['info']:
        hpwl_list.append(item['hpwl'])
    
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
        plot_dict[file_name] = {}
        plot_dict[file_name]["reward"] = evaulation_stats_all["reward"]
        plot_dict[file_name]["action"] = evaulation_stats_all["action"]
        plot_dict[file_name]['hpwl'] = hpwl_list
        
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
    model_path_list = []  # tuple of (absolute path, file name)

    for root, dirs, files in os.walk(saved_weight_path):
        for file in files:
            #if file == 'model_latest.pt':
            if file.endswith('.pt'):
                parent_dir = root.split('/')[-1]
                model_path_list.append((os.path.join(root, file), parent_dir +  "/" + file))
    print(model_path_list)
    plot_dict = {}
    plot_dict["opt"] = {}
    plot_dict["opt"]["reward"] = [10.0, 81.67260000000078, 289.1805400000003, -37.78616000000102, 171.79964000000064, 9.438239999999496, 49.610839999999826, -15.975860000000466, 37.132820000000265, 280.11299999999983, 5.284520000000157, 364.89194000000043, 135.07496000000037, 176.4788199999998, 0.0]
    plot_dict["opt"]["action"] = [49, 51, 81, 40, 60, 30, 41, 31, 19, 70, 52, 29, 62, 63, 53]
    plot_dict["opt"]["hpwl"] =  [4280.50112, 4198.828519999999, 3909.647979999999, 3947.43414, 3775.634499999999, 3766.1962599999997, 3716.58542, 3732.5612800000004, 3695.42846, 3415.3154600000003, 3410.03094, 3045.1389999999997, 2910.0640399999993, 2733.5852199999995]
    for path, filename in model_path_list:
        model = config.init_model(args.device_trainer, args.amp)
        model.load_state_dict(torch.load(path, map_location=args.device_trainer))
        test(args, config, model, log_dir, filename, plot_dict=plot_dict)
    plot_hpwl(plot_dict, args)
    plot_reward(plot_dict, args) 
    plot_action(plot_dict, args)
    return

def plot_hpwl(plot_dict, args):
    plt.figure(figsize=(10, 6))
    for name, values in plot_dict.items():
        plt.plot(values['hpwl'], label=name)
    # Adding labels and title
    plt.xlabel('action step')
    plt.ylabel('hpwl')
    plt.title(f'hpwl vs action on {args.num_target_blocks} blocks')
    plt.legend()

    # Display the plot
    # save plot
    plt.savefig('hpwl.png')
    plt.show()
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
def plot_reward(plot_dict, args):
    plt.figure(figsize=(10, 6))
    for name, values in plot_dict.items():
        plt.plot(values['reward'], label=name)
    # Adding labels and title
    plt.xlabel('action step')
    plt.ylabel('reward =(prev_hpwl-new_hpwl)')
    plt.title(f'reward vs action on {args.num_target_blocks} blocks')
    plt.legend()

    # Display the plot
    # save plot
    plt.savefig('reward_plot.png')
    plt.show()
    return

"""
Same as plot_reward but for action


"""
def plot_action(plot_dict, args):
    plt.figure(figsize=(10, 6))
    for name, values in plot_dict.items():
        plt.plot(values['action'], label=name)
    # Adding labels and title
    plt.xlabel('action step')
    plt.ylabel('action')
    plt.title(f'action vs action on {args.num_target_blocks} blocks')
    plt.legend()
    # Display the plot
    plt.savefig('action_plot.png')
    plt.show()
    return