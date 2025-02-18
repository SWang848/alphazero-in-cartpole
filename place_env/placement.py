import os
import re
import shutil
import gym
import random
from copy import deepcopy

from core.util import fill_place_file
from core.preprocess import Preprocess

import numpy as np
import pygame


class Placement(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    black = (0, 0, 0)
    white = (255, 255, 255)
    grey = (161, 161, 161)
    blue = (126, 166, 254)
    dark_blue = (126, 166, 204)
    pink = (205, 162, 190)
    orange = (255, 229, 153)

    def __init__(
        self, log_dir, simulator=False, render_mode=None, num_target_blocks=30
    ):
        # metadata = {"render.modes": ["human"]}

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # In CC, the data path saved in the local disk is set in the environment variable.
        if "data" in os.environ:
            self.data_dir = os.environ["data"]
        else:
            self.data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
            )
        self.preprocess = Preprocess(
            num_target_blocks=num_target_blocks,
            pack_xml_path=os.path.join(self.data_dir, "tseng.net"),
            block_infos_file_path=os.path.join(self.data_dir, "block.infos"),
            primitive_netlist_file_path=os.path.join(self.data_dir, "primitive.netlist"),
            grid_constraint_path=os.path.join(self.data_dir, "grid.constraint"),
            blocks_place_file_path=os.path.join(self.data_dir, "tseng.place"),
        )

        # chip information preprocess
        self.num_blocks = self.preprocess.num_target_blocks
        self.blocks_list = self.preprocess.blocks_list
        self.grid_constraints_dict = self.preprocess.grid_constraints_dict
        self.netlist_list = self.preprocess.netlist_list
        self.capacity = self.preprocess.capacity
        self.width = self.preprocess.grid_width
        self.height = self.preprocess.grid_height
        self.place_order = self.preprocess.place_order
        self.log_dir = log_dir
        self.simulator = simulator
        if self.simulator:
            # gurantee the simulator path is valid
            self.log_file_path = os.path.join(
                self.log_dir, str(random.randint(0, 9999))
            )
            if not os.path.exists(self.log_file_path):
                os.makedirs(self.log_file_path)
            place_path = os.path.join(self.data_dir, "tseng.place")
            net_path = os.path.join(self.data_dir, "tseng.net")
            shutil.copy2(place_path, os.path.join(self.log_file_path, "tseng.place"))
            shutil.copy2(net_path, os.path.join(self.log_file_path, "tseng.net"))

        # state and action space defination
        self.board_image = None
        self.place_infos = None
        self.action_space = None
        self.observation_space = None

        self.step_limit = None
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

    def step(self):
        raise NotImplementedError("subclasses must implement this method")

    def reset(self):
        raise NotImplementedError("subclasses must implement this method")

    def get_mask(self):
        raise NotImplementedError("subclasses must implement this method")

    def calculate_hpwl(self):
        hpwl_total = 0
        for i in self.netlist_list:
            x_coords, y_coords = [], []
            for j in i:
                x_coords.append(self.place_coords[j][0])
                y_coords.append(self.place_coords[j][1])
            HPWL = max(x_coords) - min(x_coords) + max(y_coords) - min(y_coords)
            if len(i) > 3:
                q = 2.7933 + 0.02616 * (len(i) - 50)
            else:
                q = 1

            HPWL = HPWL * q
            hpwl_total += HPWL
        return hpwl_total
        
    def hpwl_reward(self, hpwl):
        
        if self.num_blocks == 5:
            # 5 blocks hpwl range
            best_hpwl = 2733
            max_hpwl = 3362
        elif self.num_blocks == 15:
            # 15 blocks hpwl range
            best_hpwl = 2600
            max_hpwl = 4300
        elif self.num_blocks == 30:
            # 30 blocks hpwl range
            best_hpwl = 2600
            max_hpwl = 4900
        elif self.num_blocks == 45:
            # 45 blocks hpwl range
            best_hpwl = 2600
            max_hpwl = 5600
        elif self.num_blocks == 56:
            # 56 blocks hpwl range
            best_hpwl = 2600
            max_hpwl = 5700

        # scaled_reward = (best_hpwl_results - hpwl) / 1000
        normalized_reward = ((hpwl - best_hpwl) / (max_hpwl - best_hpwl)) * 1
        normalized_reward = max(0, min(1, normalized_reward))
        normalized_reward = -normalized_reward

        # normalized_reward = -hpwl / 1000

        return normalized_reward

    def call_simulator(self, place_coords, width):
        fill_place_file(
            place_coords,
            width,
            os.path.join(self.log_file_path, "tseng.place"),
        )
        (wire_term, critical_path_delay, wirelength) = self.episode_reward(
            self.log_file_path
        )
        return wire_term, critical_path_delay, wirelength

    def episode_reward(self, log_file_path):
        os.chdir(log_file_path)
        stream = os.popen(
            "$VTR_ROOT/vpr/vpr \
            $VTR_ROOT/vtr_flow/arch/timing/EArch.xml \
            $VTR_ROOT/vtr_flow/benchmarks/blif/tseng.blif \
            --route --route_chan_width 100 --analysis"
        )
        output = stream.read()

        wirelength = int(
            re.search(".*Total wirelength: (.*), average net length:", output).groups()[
                0
            ]
        )

        # critical_path_delay = float(
        #     re.search(".*critical path delay \(least slack\): (.*) ns,", content).groups()[0]
        # )

        # assert critical_path_delay > 0 and type(critical_path_delay) == float, "stop"
        # # Wirelength ~ 9000 to a reward ~0,higher ->better
        # critical_path_delay rescale from ~7.4 to ~7.5,higher ->better
        # sum of them
        critical_path_delay = 0
        wire_term = 0
        # print(
        #     "wire_term, critical_path_delay, wirelength",
        #     wire_term,
        #     critical_path_delay,
        #     wirelength,
        # )
        # print("wirelength", wirelength)
        return wire_term, critical_path_delay, wirelength
    
    # for mcts simulation
    def set_state(self, state):
        self = deepcopy(state)
        return self

    # for mcts simulation
    def get_state(self):
        return deepcopy(self)

    def _get_observation(self):
        raise NotImplementedError("subclasses must implement this method")

    def _place_initial_blocks(self):
        raise NotImplementedError("Subclasses must implement this method")

   
    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.render_mode == "human":
            pygame.init()
            pygame.display.init()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        self.window.fill(self.black)
        self._draw_grid()
        self._draw_info_bar()
        pygame.display.flip()

        pygame.time.delay(5000)

    def _draw_grid(self):
        for x in range(self.grid_height_size):
            for y in range(self.grid_width_size):
                matches = np.all([x, y] == self.place_coords, axis=1)
                block_index = np.where(matches)[0][0] if np.any(matches) else None

                rect = pygame.Rect(
                    y * self.square_size,
                    x * self.square_size,
                    self.square_size,
                    self.square_size,
                )
                pygame.draw.rect(self.window, self.white, rect)
                pygame.draw.rect(self.window, self.black, rect, self.border_size)
                if (x * self.grid_width_size + y) in self.grid_constraints_dict["clb"]:
                    if block_index in self.place_order:
                        pygame.draw.rect(self.window, self.blue, rect)
                    else:
                        pygame.draw.rect(self.window, self.dark_blue, rect)
                    pygame.draw.rect(self.window, self.black, rect, self.border_size)
                elif (x * self.grid_width_size + y) in self.grid_constraints_dict["io"]:
                    pygame.draw.rect(self.window, self.grey, rect)
                    pygame.draw.rect(self.window, self.black, rect, self.border_size)
                elif (x * self.grid_width_size + y) in self.grid_constraints_dict[
                    "memory"
                ]:
                    pygame.draw.rect(self.window, self.pink, rect)
                    pygame.draw.rect(self.window, self.black, rect, self.border_size)
                elif (x * self.grid_width_size + y) in self.grid_constraints_dict[
                    "mult_36"
                ]:
                    pygame.draw.rect(self.window, self.orange, rect)
                    pygame.draw.rect(self.window, self.black, rect, self.border_size)
                else:
                    pygame.draw.rect(self.window, self.white, rect)

                if block_index is not None:
                    text_surf = self.font.render(f"{block_index}", True, self.black)
                    text_pos = (
                        rect.centerx - text_surf.get_width() / 2,
                        rect.centery - text_surf.get_height() / 2,
                    )
                    self.window.blit(text_surf, text_pos)

    def _draw_info_bar(self):
        rect = pygame.Rect(
            0,
            self.grid_height_size * self.square_size,
            self.window_width,
            self.info_bar_height,
        )
        pygame.draw.rect(self.window, self.grey, rect)

        info_text = f"Steps: {self.num_step_episode} | Current Block IDX: {self.place_order[(self.num_step_episode-1)%self.num_blocks] if self.num_step_episode>0 else self.place_order[0]} | HPWL: {self.calculate_hpwl()}"
        text_surf = self.font.render(info_text, True, self.black)
        self.window.blit(text_surf, (5, self.grid_height_size * self.square_size + 5))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
