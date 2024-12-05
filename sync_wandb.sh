#!/bin/bash

# Directory containing the offline run files
# GPT generated
directory="/lustre04/scratch/truonggi/alphazero-in-cartpole/wandb"

# Prefix to match files
prefix="offline-run-20241204_"

# Loop through all files in the directory with the specified prefix
for file in "$directory/$prefix"*; do
  # Check if the file exists and is a directory
  if [ -d "$file" ]; then
    # Run wandb sync command
    wandb sync "$file"
  else
    echo "No directories found with the prefix '$prefix'"
  fi
done
