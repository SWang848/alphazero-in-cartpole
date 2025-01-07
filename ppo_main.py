import gym
from gym.vector.async_vector_env import AsyncVectorEnv

from argparse import ArgumentParser
import os

import torch
import wandb
import numpy as np
import random
from datetime import datetime

from ppo.ppo_agent import PPO
import place_env
from ppo.rollout import RolloutStorage
from ppo.evaluation import evaluate

def make_env(env_name, log_dir, simulator=False, num_target_blocks=15):
    def thunk():
        env = gym.make(
            env_name,
            log_dir=log_dir,
            simulator=simulator,
            num_target_blocks=num_target_blocks
        )
        return env

    return thunk

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
if __name__ == "__main__":
    parser = ArgumentParser("PPO Place, GO")
    parser.add_argument("--env", type=str, default="Swap-v0", help="Name of environment.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--cc", action="store_true")
    parser.add_argument("--group_name", default="default", type=str)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--rollout_size", default=64, type=int, help="Number of steps per rollout.")
    parser.add_argument("--total_timesteps", default=128000, type=int, help="Total timesteps for training.")
    parser.add_argument("--evaluation_interval", default=10, type=int, help="Interval for model evaluation.")
    parser.add_argument("--save_interval", default=20, type=int, help="Interval for saving the model.")
    parser.add_argument("--num_target_blocks", default=15, type=int, help="Number of target blocks needed to place.")
    parser.add_argument("--num_envs", default=4, type=int, help="Number of environments.")
    parser.add_argument("--mini_batch_size", default=32, type=int, help="Mini-batch size.")
    parser.add_argument("--lr_a", default=1e-4, type=float, help="Learning rate for the actor.")
    parser.add_argument("--lr_c", default=1e-4, type=float, help="Learning rate for the critic.")
    parser.add_argument("--gamma", default=1.0, type=float, help="Discount factor for future rewards.")
    parser.add_argument("--lamda", default=0.95, type=float, help="Lambda for GAE (Generalized Advantage Estimation).")
    parser.add_argument("--k_epochs", default=10, type=int, help="Number of epochs for training.")
    parser.add_argument("--entropy_coef", default=5e-3, type=float, help="Entropy coefficient for exploration.")
    parser.add_argument("--epsilon", default=0.2, type=float, help="Clip range for policy updates.")

    args = parser.parse_args()
    set_seed(args.seed)
    
    sub_dir = datetime.now().strftime("%d%m%Y_%H%M")
    sub_dir = f"{args.env}_{sub_dir}_{random.randint(10, 99)}"
    # if program is run on CC, save logs to the local disk.
    if args.cc:
        log_dir = f"{os.environ['results']}/{sub_dir}"
    else:
        if args.debug:
            sub_dir = f"debug/{sub_dir}"
        if os.path.isabs(args.results_dir):
            log_dir = os.path.join(args.results_dir, sub_dir)
        else:
            log_dir = os.path.join(os.getcwd(), args.results_dir, sub_dir)
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    
    for arg, arg_val in vars(args).items():
        print(f'setting "{arg}" config entry with {arg_val}')
            
    if args.wandb and not args.debug:
        run = wandb.init(
            project="RL for chips",
            group=args.group_name,
            config=args,
        )
        
    envs = AsyncVectorEnv(
        [
            make_env(
                env_name=args.env,
                log_dir=log_dir,
                simulator=False,
                num_target_blocks=args.num_target_blocks
            )
            for i in range(args.num_envs)
        ]
    )
    
    agent = PPO(
        num_actions=envs.single_action_space.n,
        observation_shape=envs.single_observation_space['board_image'].shape,
        args=args,
    )
    rollouts = RolloutStorage(
        args,
        observation_shape=envs.single_observation_space['board_image'].shape,
        num_actions=envs.single_action_space.n,
        device=args.device,
    )
    
    num_steps = 0
    observation_, infos = envs.reset()
    
    for i in range(0, args.total_timesteps // args.rollout_size):
        for step in range(0, args.rollout_size):
            board_image, place_infos, block_index, action_mask = (
                observation_["board_image"],
                observation_["place_infos"],
                observation_["next_block"],
                infos["action_mask"],
            )
            action, action_log_prob = agent.choose_action(
                # normalize(board_image, "board_image"),
                board_image,
                action_mask,
            )

            observation_, reward, done, truncated, infos = envs.step(action)

            rollouts.insert(
                # normalize(board_image, "board_image"),
                # normalize(observation_["board_image"], "board_image"),
                board_image,
                observation_["board_image"],
                action,
                action_log_prob,
                reward,
                done,
                action_mask,
            )
            num_steps += 1
            
            if np.any(done):
                workers_index = np.where(done == True)[0]
                print(
                    "Worker {} done in {} steps, with hpwl {}, with wirelength {}, with cumulative reward {}".format(
                        "/".join([str(item) for item in workers_index]),
                        "/".join(
                            [
                                str(item)
                                for item in [
                                    infos["episode_steps"][i] for i in workers_index
                                ]
                            ]
                        ),
                        "/".join(
                            [
                                str(item)
                                for item in [
                                    round(infos["hpwl"][i], 2) for i in workers_index
                                ]
                            ]
                        ),
                        "/".join(
                            [
                                str(item)
                                for item in [
                                    infos["wirelength"][i] for i in workers_index
                                ]
                            ]
                        ),
                        "/".join(
                            [
                                str(item)
                                for item in [
                                    round(infos["cumulative_reward"][i], 2)
                                    for i in workers_index
                                ]
                            ]
                        ),
                    )
                )
        
        (
            value_loss,
            action_loss,
            dist_entropy,
            old_approx_kl,
            approx_kl,
            clipfracs,
        ) = agent.update(rollouts)
        
        print(
            "Updates {:d}, total_steps {:d}, clipfracs {:.2f}, dist_entropy {:.2f}, value_loss {:.2f}, action_loss {:.2f} \n".format(
                i,
                i * args.rollout_size,
                clipfracs,
                dist_entropy,
                value_loss,
                action_loss,
            )
        )
        if not args.debug and args.wandb:
            wandb.log(
                {
                    "value_loss": value_loss,
                    "action_loss": action_loss,
                    "dist_entropy": dist_entropy,
                    "old_approx_kl": old_approx_kl,
                    "approx_kl": approx_kl,
                    "clipfracs": clipfracs,
                },
                step=i,
            )
            
        if i % args.save_interval == 0:
            actor_model_dict = agent.actor.state_dict()
            critic_model_dict = agent.critic.state_dict()

            torch.save(
                actor_model_dict,
                os.path.join(log_dir, "{}_actor.pth").format(i),
            )
            torch.save(
                critic_model_dict,
                os.path.join(log_dir, "{}_critic.pth").format(i),
            )
        
        if i % args.evaluation_interval == 0:
            evaluation_path = os.path.join(log_dir, "evaluation")
            if not os.path.exists(evaluation_path):
                os.makedirs(evaluation_path)
                
            env = gym.make(
                args.env,
                log_dir=evaluation_path,
                simulator=True,
                num_target_blocks=args.num_target_blocks
            )

            cumulative_reward_list, steps_episode, end_hpwl, end_wirelength = evaluate(
                agent.actor, env, args.device
            )

            cumulative_reward_mean = np.mean(cumulative_reward_list)
            steps_episode_mean = np.mean(steps_episode)
            end_hpwl_mean = np.mean(end_hpwl)
            end_wirelength_mean = np.mean(end_wirelength)

            print(
                "The {}th evaluation, the mean cumulative reward is {}, the mean last hpwl/wirelength is {}/{}, the mean episode length is {}".format(
                    i // args.evaluation_interval,
                    cumulative_reward_mean,
                    end_hpwl_mean,
                    end_wirelength_mean,
                    steps_episode_mean,
                )
            )

            if not args.debug and args.wandb:
                wandb.log(
                    {
                        "evaluation_cumulative_reward_mean": cumulative_reward_mean,
                        "evaluation_last_hpwl_mean": end_hpwl_mean,
                        "steps_episode_mean": steps_episode_mean,
                    },
                    step=i,
                )
                
    if not args.debug and args.wandb:
        wandb.finish()