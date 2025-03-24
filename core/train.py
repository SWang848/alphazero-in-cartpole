import os

import time
from statistics import mean

import numpy as np
import ray
import wandb

from config.base import BaseConfig
from core.workers import RolloutWorker
from core.replay_buffer import ReplayBuffer
from core.storage import SharedStorage


def train(args, config: BaseConfig, summary_writer, log_dir):
    print("Starting training...")
    if args.cc:
        ray.init(
            address=f"{os.environ['HEAD_NODE']}:{os.environ['RAY_PORT']}",
            _node_ip_address=os.environ["HEAD_NODE"],
        )
    else:
        ray.init()
    print("Ray initialized")
    replay_buffer = ReplayBuffer.remote(config.replay_buffer_size)
    storage = SharedStorage.remote(config, args.amp)

    rollout_workers = [
        RolloutWorker.options(
            num_cpus=args.num_cpus_per_worker, num_gpus=args.num_gpus_per_worker
        ).remote(config, args.device_workers, args.amp, replay_buffer, storage)
        for _ in range(args.num_rollout_workers)
    ]

    workers = [rollout_worker.run.remote() for rollout_worker in rollout_workers]
    storage.set_start_signal.remote()

    while True:  # Wait until RolloutWorkers collected their samples
        workers_finished = ray.get(storage.get_workers_finished.remote())
        if workers_finished != args.num_rollout_workers:
            print(
                f"{workers_finished}/{args.num_rollout_workers} workers finished..."
            )
            time.sleep(10)
            continue
        break
    
    wandb_logs = ray.get(storage.pop_wandb_logs.remote())
    best_found = ray.get(storage.get_best_found.remote())
    np.savez(os.path.join(log_dir, f"best_found.npz"), 
                hpwl=best_found["hpwl"],
                reward=best_found["reward"],
                place_infos=best_found["state"].place_infos)
    if args.wandb and not args.debug:
        print(wandb_logs)
        wandb.log(
            {   
                "rollout/avg_end_of_episode_hpwl": mean(
                    wandb_logs["end_of_episode_hpwl"]
                ),
                "rollout/avg_end_of_episode_rewards": mean(
                    wandb_logs["end_of_episode_rewards"]
                ),
                "rollout/avg_end_of_episode_wirelength": mean(
                    wandb_logs["end_of_episode_wirelength"]
                ),
                "rollout/best_found_of_episode_hpwl": min(wandb_logs["best_found_of_episode_hpwl"]),
            }
        )
        
    storage.reset_workers_finished.remote()
    ray.wait(workers)
    print("evaluation finished!")
    ray.shutdown()
