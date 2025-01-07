import torch
import numpy as np


class RolloutStorage:
    def __init__(self, args, observation_shape, num_actions, device="cpu"):
        self.device = device
        self.rollout_size = args.rollout_size
        self.board_image = torch.zeros(
            (self.rollout_size, args.num_envs) + observation_shape
        ).to(self.device)
        self.board_image_ = torch.zeros(
            (self.rollout_size, args.num_envs) + observation_shape
        ).to(self.device)
        self.actions = torch.zeros((self.rollout_size, args.num_envs)).to(self.device)
        self.action_log_probs = torch.zeros((self.rollout_size, args.num_envs)).to(
            self.device
        )
        self.rewards = torch.zeros((self.rollout_size, args.num_envs)).to(self.device)
        self.done = torch.zeros((self.rollout_size, args.num_envs)).to(self.device)
        self.action_mask = torch.zeros(
            (self.rollout_size, args.num_envs, num_actions), dtype=torch.bool
        ).to(self.device)
        self.count = 0

    def insert(
        self,
        board_image: np.ndarray,
        board_image_: np.ndarray,
        actions: np.ndarray,
        action_log_probs: np.ndarray,
        rewards: np.ndarray,
        done: np.ndarray,
        action_mask: np.ndarray,
    ):
        self.board_image[self.count] = torch.tensor(board_image).to(self.device)
        self.board_image_[self.count] = torch.tensor(board_image_).to(self.device)
        self.actions[self.count] = torch.tensor(actions).to(self.device)
        self.action_log_probs[self.count] = torch.tensor(action_log_probs).to(
            self.device
        )
        self.rewards[self.count] = torch.tensor(rewards).to(self.device)
        self.done[self.count] = torch.tensor(done, dtype=torch.bool).to(self.device)
        self.action_mask[self.count] = torch.tensor(np.stack(action_mask, axis=0), dtype=torch.bool).to(
            self.device
        )
        self.count = (self.count + 1) % self.rollout_size
