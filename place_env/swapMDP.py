import os
import re
import shutil
import gym
import random
from gym import spaces
from copy import deepcopy

from core.util import fill_place_file, trans_coordinate
from core.preprocess import Preprocess
from place_env.placement import Placement

import numpy as np
import pygame


class SwapPlacement(Placement):

    def __init__(
        self, log_dir, simulator=False, render_mode=None, num_target_blocks=30
    ):
        super().__init__(log_dir, simulator, render_mode, num_target_blocks)
        self.prev_actions = []
        self.cheat_trajectory = self.cheat(file_path=os.path.join(self.data_dir, "optimized.place"))

        # state and action space defination
        self.board_image = np.zeros((2, self.width, self.height), dtype=int)
        self.place_infos = np.full((len(self.blocks_list), 7), -1)
        self.action_space = spaces.Discrete(self.width * self.height)
        self.observation_space = spaces.Dict(
            {
                "board_image": spaces.Box(
                    low=0,
                    high=np.inf,
                    shape=(
                        6,
                        self.height,
                        self.width,
                    ),  # capacity, current block, swap_place, sink, source, connections
                    dtype=float,
                ),
                "place_infos": spaces.Box(
                    low=-1, high=np.inf, shape=(len(self.blocks_list), 7), dtype=float
                ),
                "next_block": spaces.Discrete(self.num_blocks),
            }
        )

        self.episode_step_limit = self.preprocess.num_target_blocks
        self.num_step = 0
        self.num_episode = 0
        self.num_step_episode = 0
        self.cumulative_reward = 0
        self.place_coords = np.full((len(self.blocks_list), 2), -1)

        # render
        self.grid_width_size = self.width
        self.grid_height_size = self.height
        self.square_size = 40
        self.border_size = 1
        self.info_bar_height = 50
        self.window_width = self.grid_width_size * self.square_size
        self.window_height = (
            self.grid_height_size * self.square_size + self.info_bar_height
        )
        if render_mode is None:
            self.window = None
            self.font = None
        else:
            pygame.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            self.font = pygame.font.Font(None, 24)

    def step(self, action):
        x = action // self.width
        y = action % self.width
        self.action = action
        self.prev_actions.append(self.action)

        # last_hpwl = self.calculate_hpwl()
        block_index = self.place_order[self.num_step_episode % self.num_blocks]
        board_image, place_infos = self._get_observation(block_index, x, y)

        hpwl = self.calculate_hpwl()
        reward = self.hpwl_reward(hpwl)
        # reward = self.hpwl_diff_reward(last_hpwl, hpwl)
        # if action == self.cheat_trajectory[self.num_step_episode]:
        #     reward = 0.0
        done = False
        wirelength = 0
        truncated = False

        if (self.num_step_episode == self.episode_step_limit - 1):
            done = True
            self.num_episode += 1
            if self.simulator:
                (wire_term, critical_path_delay, wirelength) = self.call_simulator(
                    self.place_coords, self.width
                )
            else:
                (wire_term, critical_path_delay, wirelength) = (0, 0, 0)
        
        self.cumulative_reward += 1**self.num_step_episode * reward
        self.num_step += 1
        self.num_step_episode += 1
        action_mask = self.get_mask()
        next_block = self.place_order[self.num_step_episode % self.num_blocks]

        infos = {
            "placed_block": block_index,
            "hpwl": hpwl,
            "episode_steps": self.num_step_episode,
            "cumulative_reward": self.cumulative_reward,
            "wirelength": wirelength,
            "num_episode": self.num_episode,
            "action_mask": action_mask,
        }
        
        return (
            {
                "board_image": board_image,
                "place_infos": place_infos,
                "next_block": next_block,
            },
            reward,
            done,
            truncated,
            infos,
        )

    def reset(self):
        self.num_step_episode = 0
        self.cumulative_reward = 0
        self.prev_actions = []
        self.board_image, self.place_infos, self.place_coords = self._place_initial_blocks(
                optimized_file=os.path.join(self.data_dir, "optimized.place")
            )
        
        hpwl = self.calculate_hpwl()

        if self.simulator:
            (wire_term, critical_path_delay, wirelength) = self.call_simulator(
                self.place_coords, self.width
            )
        else:
            (wire_term, critical_path_delay, wirelength) = (0, 0, 0)

        
        infos = {
            "placed_block": None,
            "hpwl": hpwl,
            "episode_steps": self.num_step_episode,
            "cumulative_reward": self.cumulative_reward,
            "wirelength": wirelength,
            "num_episode": self.num_episode,
            "action_mask": self.get_mask(),
        }

        return {
            "board_image": self.board_image,
            "place_infos": self.place_infos,
            "next_block": self.place_order[0],
        }, infos

    def get_mask(self, block_index=None):
        if block_index is None:
            block_index = self.place_order[
                self.num_step_episode % self.num_blocks
            ]

        block_type = self.blocks_list.loc[self.blocks_list["index"] == block_index][
            "type"
        ].values[0]
        valid_positions = self.grid_constraints_dict[block_type].copy()

        for i, position in enumerate(self.place_coords):
            if i in self.place_order:
                pass
            elif (
                i not in self.place_order
                and (position[0] * self.width + position[1]) in valid_positions
            ):
                valid_positions.remove(position[0] * self.width + position[1])
            
            if (position[0] * self.width + position[1]) in self.prev_actions:
                valid_positions.remove(position[0] * self.width + position[1])

        action_mask = np.zeros((self.height * self.width), dtype=int)
        action_mask[valid_positions] = 1

        return action_mask

    def cheat(self, file_path):
        optimized_action = list()
        with open(file_path, "r") as file:
            for index, line in enumerate(file.readlines()):
                if 60 >= index >= 5:
                    line_split = line.strip().split()
                    coord = trans_coordinate([int(line_split[1]), int(line_split[2])], 11, "cs")
                    optimized_action.append(coord[0] * 11 + coord[1])
        
        return optimized_action

    def hpwl_diff_reward(self, last_hpwl, hpwl):
        reward = (last_hpwl - hpwl)/10
        return reward
    
    def _get_observation(self, block_index, coord_x, coord_y):

        current_block_coord_x = self.place_coords[block_index][0]
        current_block_coord_y = self.place_coords[block_index][1]

        next_block_coord_x = self.place_coords[
            self.place_order[(self.num_step_episode + 1) % self.num_blocks]
        ][0]
        next_block_coord_y = self.place_coords[
            self.place_order[(self.num_step_episode + 1) % self.num_blocks]
        ][1]

        # place block to empty grid
        if self.board_image[0, coord_x, coord_y] == 0:
            # place_info update
            self.place_infos[block_index][1:3] = [coord_x, coord_y]
            self.place_coords[block_index] = [coord_x, coord_y]

            # board_image update
            self.board_image[[0, 3, 4, 5], coord_x, coord_y] = self.board_image[
                [0, 3, 4, 5], current_block_coord_x, current_block_coord_y
            ]
            self.board_image[
                [0, 1, 3, 4, 5], current_block_coord_x, current_block_coord_y
            ] = 0

            self.board_image[1, next_block_coord_x, next_block_coord_y] = 1

        # swap block
        elif self.board_image[0, coord_x, coord_y] != 0:
            # place_info update
            swap_block_index = int(
                np.where(np.all(self.place_coords == [coord_x, coord_y], axis=1))[0]
            )
            self.place_infos[block_index][1:3] = [coord_x, coord_y]
            self.place_infos[swap_block_index][1:3] = [
                current_block_coord_x,
                current_block_coord_y,
            ]
            self.place_coords[block_index] = [coord_x, coord_y]
            self.place_coords[swap_block_index] = [
                current_block_coord_x,
                current_block_coord_y,
            ]

            # board_image update
            swap_board_value = self.board_image[[0, 3, 4, 5], coord_x, coord_y]
            self.board_image[[0, 3, 4, 5], coord_x, coord_y] = self.board_image[
                [0, 3, 4, 5], current_block_coord_x, current_block_coord_y
            ]
            self.board_image[
                [0, 3, 4, 5], current_block_coord_x, current_block_coord_y
            ] = swap_board_value

            self.board_image[1, current_block_coord_x, current_block_coord_y] = 0
            self.board_image[1, next_block_coord_x, next_block_coord_y] = 1

        return self.board_image.copy(), self.place_infos.copy()

    def _place_initial_blocks(self, optimized_file, seed=0):
        """CXB experiment"""
        place_coords = np.full((len(self.blocks_list), 2), -1)
        board_image = np.zeros(
            self.observation_space["board_image"].shape, dtype=float
        )
        place_infos = np.full(
            self.observation_space["place_infos"].shape, -1, dtype=int
        )
        # place the initial blocks
        swappable_positions = [[], []]
        valid_positions = self.grid_constraints_dict["clb"].copy()
        with open(optimized_file, "r") as file:
            for index, line in enumerate(file.readlines()):
                if index >= self.num_blocks + 5:
                    line_split = line.strip().split()

                    # coordinates translation
                    coords = trans_coordinate(
                        [int(line_split[1]), int(line_split[2])], self.width, "cs"
                    )
                    x, y = coords[0], coords[1]
                    block_index = int(line_split[-1][1:])

                    place_coords[block_index] = [x, y]
                    board_image[0, x, y] += 1
                    num_sink = self.blocks_list.loc[
                        self.blocks_list["index"] == block_index
                    ]["sink"].values[0]
                    num_source = self.blocks_list.loc[
                        self.blocks_list["index"] == block_index
                    ]["source"].values[0]
                    num_connections = self.blocks_list.loc[
                        self.blocks_list["index"] == block_index
                    ]["connections"].values[0]
                    board_image[3, x, y] = num_sink
                    board_image[4, x, y] = num_source
                    board_image[5, x, y] = num_connections
                    type = self.blocks_list.loc[
                        self.blocks_list["index"] == block_index
                    ]["type"].values[0]

                    place_infos[block_index] = [
                        block_index,
                        x,
                        y,
                        num_source,
                        num_sink,
                        num_connections,
                        0 if type == "clb" else 1,
                    ]
                    if type == "clb":
                        valid_positions.remove(x * self.width + y)
        for i in valid_positions:
            swappable_positions[0].append(i // self.width)
            swappable_positions[1].append(i % self.width)

        # random place the target blocks
        # np.random.seed(seed)
        valid_position = self.get_mask()
        for block_index in self.place_order:
            random_index = np.random.choice(np.where(valid_position == 1)[0])
            x, y = random_index // self.width, random_index % self.width

            place_coords[block_index] = [x, y]
            board_image[0, x, y] += 1
            if block_index == self.place_order[0]:
                board_image[1, x, y] = 1

            num_sink = self.blocks_list.loc[self.blocks_list["index"] == block_index][
                "sink"
            ].values[0]
            num_source = self.blocks_list.loc[self.blocks_list["index"] == block_index][
                "source"
            ].values[0]
            num_connections = self.blocks_list.loc[
                self.blocks_list["index"] == block_index
            ]["connections"].values[0]
            board_image[3, x, y] = num_sink
            board_image[4, x, y] = num_source
            board_image[5, x, y] = num_connections
            type = self.blocks_list.loc[self.blocks_list["index"] == block_index][
                "type"
            ].values[0]

            place_infos[block_index] = [
                block_index,
                x,
                y,
                num_source,
                num_sink,
                num_connections,
                0 if type == "clb" else 1,
            ]

            valid_position[random_index] = 0
        board_image[2, swappable_positions[0], swappable_positions[1]] = 1
        # print(self.calculate_hpwl())
        # print(self.calculate_hpwl())

        return (
            board_image,
            place_infos,
            place_coords,
        )