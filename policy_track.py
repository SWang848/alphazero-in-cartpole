from argparse import ArgumentParser
import os
import ray
import torch
import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from config.place import Config
from core.storage import add_logs
from core.workers import TestWorker

import logging

logging.getLogger("ray").setLevel(logging.ERROR)


def policy_track(args, config, model):
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
    print(f"action:{action_trajectory}, reward:{reward_trajectory}")
    print(
        f"the best hpwl is: {best_found['hpwl']}, the best reward is: {best_found['reward']}"
    )

    for worker in test_workers:
        ray.kill(worker)

    return best_found["hpwl"]


if __name__ == "__main__":
    parser = ArgumentParser("MCTS Place, GO")
    parser.add_argument("--env", type=str, default="Swap-v0")
    # parser.add_argument("--env", type=str, default="Classic-v0")
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--opr", default="train", type=str)
    parser.add_argument("--num_rollout_workers", default=1, type=int)
    parser.add_argument("--num_cpus_per_worker", default=16, type=float)
    parser.add_argument("--num_gpus_per_worker", default=1, type=float)
    parser.add_argument("--num_test_episodes", default=1, type=float)
    parser.add_argument("--model_path", default=None)
    # parser.add_argument(
    #     "--model_dir",
    #     # default="/home/swang848/efficientalphazero/results/Swap-v0_24022025_1328_59"
    #     # default="/home/swang848/efficientalphazero/results/Swap-v0_02032025_2138_59",
    #     # default="/home/swang848/efficientalphazero/results/Swap-v0_05032025_1344_59",
    #     # default="/home/swang848/efficientalphazero/results/Swap-v0_10032025_1656_59",
    #     # default="/home/swang848/efficientalphazero/results/results_cc/Swap-v0_24032025_1809_93", #c15b_forced_exploration_fixed_init_2
    #     # default="/home/swang848/efficientalphazero/results/results_cc/Swap-v0_24032025_1821_37"  #c15b_forced_exploration_non_fixed_init_2
    #     # default="/home/swang848/efficientalphazero/results/results_cc/c15b_fixed_init_2"  #c15b_fixed_init_2
    #     # default="/home/swang848/efficientalphazero/results/results_cc/Swap-v0_25032025_0422_93" #c15b_non_fixed_init_2
    #     default="/home/swang848/efficientalphazero/results_CC/c15b_gumbel_cc_NFI_0_776",
    # )
    parser.add_argument(
        "--model_dir_pool",
        nargs="+", 
        default=["/home/swang848/efficientalphazero/results_CC/c15b_gumbel_cc_0_776",
                #  "/home/swang848/efficientalphazero/results_CC/c15b_gumbel_cc_12_275",
                 "/home/swang848/efficientalphazero/results_CC/c15b_gumbel_cc_529_762",
                 "/home/swang848/efficientalphazero/results_CC/c15b_gumbel_cc_723_179",
                 "/home/swang848/efficientalphazero/results_CC/c15b_gumbel_cc_8764_960"],
        help="List of model directories to evaluate"
    )
    parser.add_argument("--device_workers", default="cuda", type=str)
    parser.add_argument("--device_trainer", default="cuda", type=str)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", default=33554432, type=int)
    parser.add_argument("--non_fixed_init", action="store_true")
    parser.add_argument("--num_target_blocks", default=15, type=int)
    parser.add_argument("--c_init", default=2.5, type=float)
    parser.add_argument("--num_simulations", default=150, type=int)
    parser.add_argument("--num_envs_per_worker", default=1, type=int)
    parser.add_argument("--value_support_min", default=-10, type=int)
    parser.add_argument("--value_support_max", default=0, type=int)
    parser.add_argument("--value_support_delta", default=1, type=int)
    parser.add_argument("--m_top", default=8, type=int)
    parser.add_argument("--c_visit", default=36, type=int)
    parser.add_argument("--c_scale", default=0.1, type=int)

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

    ray.init(log_to_driver=False)
    all_results = {}
    for model_dir in args.model_dir_pool:
        print(f"\nEvaluating models in directory: {model_dir}")
        # sort model paths by training steps in ascending order
        model_paths = sorted(
            [
                os.path.join(model_dir, f)
                for f in os.listdir(model_dir)
                if f.endswith(".pt") and "best" not in f and "latest" not in f
            ],
            key=lambda x: int(os.path.basename(x).split(".")[0].split("_")[-1]),
            reverse=False,
        )
        latest_model = os.path.join(model_dir, "model_latest.pt")
        if os.path.exists(latest_model):
            model_paths.append(latest_model)
            
        best_hpwl_list = []
        for model_path in model_paths:
            setattr(config, "model_path", model_path)
            model.load_state_dict(torch.load(model_path))
            best_hpwl = policy_track(args, config, model)
            best_hpwl_list.append(best_hpwl)
            
        model_steps = [
            str(os.path.basename(path).split(".")[0].split("_")[-1]) for path in model_paths
        ]
        
        # Store results for this directory
        all_results[model_dir] = {
            'model_steps': model_steps,
            'best_hpwl_list': best_hpwl_list
        }
        
       
    # plt.figure(figsize=(12, 6))
    # plt.plot(model_steps, best_hpwl_list, marker='o', linestyle='-', linewidth=2, markersize=8)

    # Create a single plot for all model directories
    plt.figure(figsize=(15, 8))
    
    # Define colors and markers for different model directories
    colors = plt.cm.tab10(np.linspace(0, 1, len(args.model_dir_pool)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, (model_dir, results) in enumerate(all_results.items()):
        model_steps = results['model_steps']
        best_hpwl_list = results['best_hpwl_list']
        
        # Create x-axis positions for plotting
        x_positions = list(range(len(model_steps)))
        
        # Plot with unique color and marker for each model directory
        plt.plot(x_positions, best_hpwl_list, 
                marker=markers[idx % len(markers)],
                linestyle='-',
                linewidth=2,
                markersize=8,
                color=colors[idx],
                # label=os.path.basename(model_dir)
                )
        
        # Add annotations for each point
        for x_pos, step, hpwl in zip(x_positions, model_steps, best_hpwl_list):
            plt.annotate(f'{hpwl:.2f}',
                        (x_pos, hpwl),
                        textcoords="offset points",
                        xytext=(0,5),
                        ha='center',
                        fontsize=8)

    plt.title('Best HPWL vs Training Steps for Gumbel Models in 15b', fontsize=14)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Best HPWL', fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis ticks to show actual step values
    plt.xticks(x_positions, model_steps, rotation=45)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()

    # Save the plot in the current directory
    plot_path = "combined_model_comparison.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()

    ray.shutdown()
    print("Finished!")
