
import os
import shutil
from place_env.placement import Placement
from ppo import PPO
import gym
import torch
import numpy as np
from datetime import datetime
import random
from torch.distributions import Categorical
from model import Actor, Critic

class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, masks=None, device="cpu"):
        self.original_logits = logits.clone()
        self.device = device
        self.masks = masks
        if masks is None:
            super(CategoricalMasked, self).__init__(probs=probs, logits=logits)
        else:
            logits = torch.where(self.masks, logits, torch.tensor(-float('inf')).to(self.device))
            super(CategoricalMasked, self).__init__(probs=probs, logits=logits)

    def entropy(self):
        if self.masks is None:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(self.device))
        return -p_log_p.sum(-1)

def choose_action(actor, board_image, action_mask, device):
    with torch.no_grad():
        board_image = torch.tensor(board_image, dtype=torch.float32).to(device)
        logits = actor(board_image)

        dist = CategoricalMasked(
            logits=logits,
            masks=action_mask,
            device=device,
        )
        a = dist.sample()
        # a = dist.probs.argmax(dim=-1)

    return a.cpu().numpy()


def calculate_critic_loss(reward_list, done_list, value_list):
    lastgae = 0
    gamma = 0.99
    lamda = 0.95
    adv = np.zeros_like(reward_list)

    with torch.no_grad():
        for t in reversed(range(len(reward_list))):
            if t == len(reward_list) - 1:
                nextnonterminal = 1.0
                nextvalues = 0
            else:
                nextnonterminal = 1.0 - done_list[t + 1]
                nextvalues = value_list[t + 1]
            delta = (
                reward_list[t] + gamma * nextvalues * nextnonterminal - value_list[t]
            )

            adv[t] = lastgae = delta + gamma * lamda * lastgae * nextnonterminal
        returns = adv + value_list
        v_loss = np.mean((returns - value_list) ** 2)
    return adv, returns, v_loss


def evaluate(actor, env, device, critic=None):
    done = False
    observation_ = env.reset()
    cumulative_reward_list = []
    episode_steps = []
    end_hpwl = []
    end_wirelength = []

    reward_list = []
    done_list = []
    value_list = []
    for j in range(10):
        i = 0
        cumulative_reward = 0
        while not done:
            board_image, place_infos, block_index, action_mask = (
                observation_["board_image"],
                observation_["place_infos"],
                observation_["next_block"],
                observation_["action_mask"],
            )
            action = choose_action(
                actor,
                np.expand_dims(board_image, 0),
                np.expand_dims(action_mask, 0),
                device,
            )
            if critic is not None:
                value = critic(
                    torch.tensor(
                        np.expand_dims(board_image, 0),
                        dtype=torch.float32,
                    ).to(device)
                )
                # print(value)
                value_list.append(value.item())
            observation_, reward, done, infos = env.step(action[0])
            if critic is not None:
                reward_list.append(reward)
                done_list.append(done)
            env.render() if env.render_mode == "human" else None
            cumulative_reward += 1**i * reward
            i = i + 1
        episode_steps.append(infos["episode_steps"])
        end_hpwl.append(infos["hpwl"])
        end_wirelength.append(infos["wirelength"])
        cumulative_reward_list.append(cumulative_reward)
        done = False
    if critic is not None:
        adv, returns, v_loss = calculate_critic_loss(reward_list, done_list, value_list)
    env.close()
    return cumulative_reward_list, episode_steps, end_hpwl, end_wirelength
    