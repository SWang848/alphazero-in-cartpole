import numpy as np

# Load the .npz file
data = np.load("/home/swang848/efficientalphazero/results/non_fixed_init/best_found.npz")

# List the contents (keys) of the file
print(data.files)

# Access data by key
for key in data.files:
    print(f"{key}: {data[key]}")