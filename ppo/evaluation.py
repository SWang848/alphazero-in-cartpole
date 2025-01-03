from preprocess import Preprocess
import os
import shutil
from place_env.placement import Placement
from ppo import PPO
import gym
from utils import build_graph, normalize
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
            logits = torch.where(self.masks, logits, torch.tensor(-1e8).to(device))
            super(CategoricalMasked, self).__init__(probs=probs, logits=logits)

    def entropy(self):
        if self.masks is None:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(self.device))
        return -p_log_p.sum(-1)

    # for vis probs
    def get_unmasked_probs(self):
        return torch.nn.functional.softmax(self.original_logits, dim=-1)


def choose_action(actor, board_image, place_infos, blocks_index, action_mask, device):
    with torch.no_grad():
        board_image = torch.tensor(board_image, dtype=torch.float32).to(device)
        place_infos = torch.tensor(place_infos, dtype=torch.float).to(device)
        action_mask = torch.tensor(action_mask, dtype=torch.bool).to(device)
        logits = actor(board_image, place_infos, blocks_index)

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
                normalize(np.expand_dims(board_image, 0), "board_image"),
                normalize(np.expand_dims(place_infos, 0), "place_infos"),
                np.expand_dims(block_index, 0),
                np.expand_dims(action_mask, 0),
                device,
            )
            if critic is not None:
                value = critic(
                    torch.tensor(
                        normalize(np.expand_dims(board_image, 0), "board_image"),
                        dtype=torch.float32,
                    ).to(device),
                    torch.tensor(
                        normalize(np.expand_dims(place_infos, 0), "place_infos"),
                        dtype=torch.float,
                    ).to(device),
                    np.expand_dims(block_index, 0),
                )
                # print(value)
                value_list.append(value.item())
            observation_, reward, done, infos = env.step(action[0])
            if critic is not None:
                reward_list.append(reward)
                done_list.append(done)
            env.render() if env.render_mode == "human" else None
            cumulative_reward += 0.99**i * reward
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


if __name__ == "__main__":
    eda_root = os.getcwd()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preprocess = Preprocess(
        5,
        os.path.join(eda_root, "data/tseng.net"),
        os.path.join(eda_root, "data/block.infos"),
        os.path.join(eda_root, "data/primitive.netlist"),
        os.path.join(eda_root, "data/grid.constraint"),
        os.path.join(eda_root, "data/tseng.place"),
    )

    folder_name = (
        f"{datetime.today().strftime('%Y%m%d%H%M%S')}_{random.randint(1000, 9999)}"
    )
    evaluation_path = os.path.join(eda_root, "evaluation", folder_name)
    if not os.path.exists(evaluation_path):
        os.makedirs(evaluation_path)
    circuit_data_root = os.path.join(eda_root, "data/")
    place_path = os.path.join(circuit_data_root, "tseng.place")
    net_path = os.path.join(circuit_data_root, "tseng.net")
    shutil.copy2(place_path, os.path.join(evaluation_path, "tseng_empty.place"))
    shutil.copy2(net_path, evaluation_path)
    actor_weights_path = "/home/swang848/RL-FPGA/model_weights/700_actor.pth"
    critic_weights_path = "/home/swang848/RL-FPGA/model_weights/700_critic.pth"

    env = gym.make(
        "place_env-v0",
        preprocess=preprocess,
        truncate_step=50,
        simulator_path=evaluation_path,
        render_mode="human",
    )
    netlist_graph = build_graph(preprocess.netlist_list).to(device)

    actor = Actor(
        netlist_graph=netlist_graph,
        action_dim=env.action_space.n,
        target_blocks_index=env.place_order,
    ).to(device)
    actor.load_state_dict(torch.load(actor_weights_path, map_location=device))

    critic = Critic(
        netlist_graph=netlist_graph,
        target_blocks_index=env.place_order,
    ).to(device)
    critic.load_state_dict(torch.load(critic_weights_path, map_location=device))

    cumulative_reward_list, episode_steps, end_hpwl, end_wirelength = evaluate(
        actor, env, device, critic
    )
