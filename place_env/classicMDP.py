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


class ClassicPlacement(Placement):

    def __init__(
        self, log_dir, simulator=False, render_mode=None, num_target_blocks=30
    ):
        super().__init__(log_dir, simulator, render_mode, num_target_blocks)
        # state and action space defination
        self.board_image = np.zeros((6, self.width, self.height), dtype=int)
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
                    ),  # capacity, wirelength, availiable_grids, sink, source, connections
                    dtype=float,
                ),
                "place_infos": spaces.Box(
                    low=-1, high=np.inf, shape=(len(self.blocks_list), 7), dtype=float
                ),
                "next_block": spaces.Discrete(self.num_blocks),
            }
        )
        

        self.episode_step_limit = self.preprocess.num_target_blocks
        self.init_board_image, self.init_place_infos, self.init_place_coords = (
            self._place_initial_blocks(
                optimized_file=os.path.join(self.data_dir, "optimized.place")
            )
        )
        
    def step(self, action):
        x = action // self.width
        y = action % self.width
        self.action = action
        
        block_index = self.place_order[self.num_step_episode % self.num_blocks]
        board_image, place_infos = self._get_observation(block_index, x, y)
        
        reward = 0
        hpwl = 0
        done = False
        wirelength = 0
        truncated = False
        
        if (self.num_step_episode == self.episode_step_limit - 1):
            done = True
            self.num_episode += 1
            hpwl = self.calculate_hpwl()
            reward = self.hpwl_reward(hpwl)
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
        self.place_coords = self.init_place_coords.copy()
        self.board_image = self.init_board_image.copy()
        self.place_infos = self.init_place_infos.copy()


        (wire_term, critical_path_delay, wirelength) = (0, 0, 0)

        infos = {
            "placed_block": None,
            "hpwl": 0,
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
            if ( 
                np.all(position != -1)
                and (position[0] * self.width + position[1]) in valid_positions
            ):
                valid_positions.remove(position[0] * self.width + position[1])

        action_mask = np.zeros((self.height * self.width), dtype=int)
        action_mask[valid_positions] = 1
        
        return action_mask
    
    def _get_observation(self, block_index, coord_x, coord_y):
        self.place_infos[block_index][1:3] = [coord_x, coord_y]
        self.place_coords[block_index] = [coord_x, coord_y]
        
        num_sink = self.blocks_list.loc[
            self.blocks_list["index"] == block_index
        ]["sink"].values[0]
        num_source = self.blocks_list.loc[
            self.blocks_list["index"] == block_index
        ]["source"].values[0]
        num_connections = self.blocks_list.loc[
            self.blocks_list["index"] == block_index
        ]["connections"].values[0]
        self.board_image[0] += 1
        self.board_image[2, coord_x, coord_y] = 0
        self.board_image[3, coord_x, coord_y] = num_sink
        self.board_image[4, coord_x, coord_y] = num_source
        self.board_image[5, coord_x, coord_y] = num_connections
        
        next_block_idx = self.place_order[(self.num_step_episode + 1) % self.num_blocks]
        # the wire-mask channel
        for net in self.netlist_list:
            coords_x = []
            coords_y = []
            if next_block_idx in net:
                for node in net:
                    if np.all(self.place_coords[node] == -1):
                        pass
                    else:
                        coords_x.append(self.place_coords[node][0])
                        coords_y.append(self.place_coords[node][1])

                if len(coords_x) == 0:
                    min_x = 0
                    min_y = 0
                    max_x = self.height - 1
                    max_y = self.width - 1
                else:
                    min_x = min(coords_x)
                    max_x = max(coords_x)
                    min_y = min(coords_y)
                    max_y = max(coords_y)

                if len(net) > 3:
                    q = 2.7933 + 0.02616 * (len(net) - 50)
                else:
                    q = 1

                for i in range(self.height):
                    if i >= min_x and i <= max_x:
                        pass
                    elif i < min_x:
                        self.board_image[1, i, :] = self.board_image[1, i, :] + q * (min_x - i)
                    elif i > max_x:
                        self.board_image[1, i, :] = self.board_image[1, i, :] + q * (i - max_x)

                for j in range(self.width):
                    if j >= min_y and j <= max_y:
                        pass
                    elif j < min_y:
                        self.board_image[1, :, j] = self.board_image[1, :, j] + q * (min_y - j)
                    elif j > max_y:
                        self.board_image[1, :, j] = self.board_image[1, :, j] + q * (j - max_y)
            else:
                pass
        
        return self.board_image.copy(), self.place_infos.copy()
    
    def _place_initial_blocks(self, optimized_file, seed=0):
        """CXB experiment"""
        place_coords = np.full((len(self.blocks_list), 2), -1)
        board_image = np.zeros(
            self.observation_space["board_image"].shape, dtype=float
        )
        board_image[2, :, :] = 1
        place_infos = np.full(
            self.observation_space["place_infos"].shape, -1, dtype=int
        )
        # place the initial blocks
        
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
                    board_image[2, x, y] = 0
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
        
        # wiremask channel
        first_block = self.place_order[0]
        for net in self.netlist_list:
            coords_x = []
            coords_y = []
            if first_block in net:
                for node in net:
                    if np.all(place_coords[node] == -1):
                        pass
                    else:
                        coords_x.append(place_coords[node][0])
                        coords_y.append(place_coords[node][1])

                if len(coords_x) == 0:
                    min_x = 0
                    min_y = 0
                    max_x = self.height - 1
                    max_y = self.width - 1
                else:
                    min_x = min(coords_x)
                    max_x = max(coords_x)
                    min_y = min(coords_y)
                    max_y = max(coords_y)

                if len(net) > 3:
                    q = 2.7933 + 0.02616 * (len(net) - 50)
                else:
                    q = 1

                for i in range(self.height):
                    if i >= min_x and i <= max_x:
                        pass
                    elif i < min_x:
                        board_image[1, i, :] = board_image[1, i, :] + q * (min_x - i)
                    elif i > max_x:
                        board_image[1, i, :] = board_image[1, i, :] + q * (i - max_x)

                for j in range(self.width):
                    if j >= min_y and j <= max_y:
                        pass
                    elif j < min_y:
                        board_image[1, :, j] = board_image[1, :, j] + q * (min_y - j)
                    elif j > max_y:
                        board_image[1, :, j] = board_image[1, :, j] + q * (j - max_y)
            else:
                pass

        return (
            board_image,
            place_infos,
            place_coords,
        )
