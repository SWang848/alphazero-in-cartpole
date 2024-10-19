import place_env
import gym

import os

from config.place import Config

EDA_ROOT = os.environ.get("EDA_ROOT")
env = gym.make("Place-v0", log_dir=None, simulator=False, num_target_blocks=5, render_mode="human")
obs, info = env.reset()  # obs is dictionary 

# board height
print("board height: ", env.height)  # 11
print("board width: ", env.width)    # 11
print("obs.board_image shape: ", obs['board_image'].shape)  # (6, height, width)
print("self.block_list length: ", len(env.blocks_list))  # 230
print("num_blocks  : ", env.num_blocks)  # 5


# config
config = Config()
print("config.obs_shape", config.obs_shape)  # (6, 11, 11)
print("config.obs_shape[0]", config.obs_shape[0])  # 230

# MCTSrolloutwintdows. its obs is stacking of 5 frames of obs_shape
from core.replay_buffer import TransitionBuffer, ReplayBuffer, MCTSRollingWindow
mctsrolloutwindows = MCTSRollingWindow(config.obs_shape, config.frame_stack)  # obs_shape = (6, 11, 11), frame_stack = 5
print("rollout obs.shape", mctsrolloutwindows.obs.shape)  # (30, 11, 11) obs = np.zeros((self.obs_shape[0] * self.frame_stack, *self.obs_shape[1:])
print("*obs_shape[1: ]", *config.obs_shape[1:])  # 11 11


