import numpy as np

# Load the .npz file
# data = np.load("/home/swang848/efficientalphazero/results/non_fixed_init_longrun/Swap-v0_10032025_1933_59/best_found.npz")
data = np.load("/home/swang848/efficientalphazero/results/non_fixed_init_longrun/best_found.npz")
# List the contents (keys) of the file
print(data.files)

# Access data by key
for key in data.files:
    print(f"{key}: {data[key]}")