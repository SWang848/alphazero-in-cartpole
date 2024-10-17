import place_env
import gym

import os
# pwd
path_to_file = os.path.abspath(__file__)
path_ot_folder = os.path.dirname(path_to_file)

EDA_ROOT = os.environ.get("EDA_ROOT")
print("EDA_ROOT: ", EDA_ROOT)
print("path_ot_folder: ", path_ot_folder)

env = gym.make("Place-v0", log_dir=None, simulator=False, render_mode="human")
obs, info = env.reset()

env.unwrapped.render()
print(info)