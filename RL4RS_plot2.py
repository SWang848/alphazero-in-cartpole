import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from datetime import datetime
import seaborn as sns

# Set the style for better-looking plots with white background
sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white"})

# Define a modern color palette
COLORS = {
    'gumbel': '#2E86C1',  # Soft blue
    'mcts': '#E91E63',    # Pink
    'ppo': '#00BCD4',     # Teal
    'vtr': '#FF9800'      # Orange
}

# Load data from CSV files
script_dir = os.path.dirname(os.path.abspath(__file__))

# Function to load data from a specific directory
def load_data(directory):
    csv_files = glob.glob(os.path.join(script_dir, directory, '*.csv'))
    timestamps = []
    values = []
    
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            # Normalize timestamps to start from 0
            time_values = df[df.columns[0]].values
            time_values = time_values - time_values[0]  # Subtract the first value to start from 0
            timestamps.append(time_values)
            values.append(df[df.columns[1]].values)
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue
    return timestamps, values

# Load all datasets for 5 blocks
gumbel_timestamps_5, gumbel_values_5 = load_data('RL4RS/gumbel5')
mcts_timestamps_5, mcts_values_5 = load_data('RL4RS/mcts5')
ppo_timestamps_5, ppo_values_5 = load_data('RL4RS/ppo5')

# Load all datasets for 15 blocks
gumbel_timestamps_15, gumbel_values_15 = load_data('RL4RS/gumbel15')
mcts_timestamps_15, mcts_values_15 = load_data('RL4RS/mcts15')
ppo_timestamps_15, ppo_values_15 = load_data('RL4RS/ppo15')

# Function to truncate data to end at a specific timestamp
def truncate_to_time(timestamps, values, end_time):
    truncated_timestamps = []
    truncated_values = []
    for ts, v in zip(timestamps, values):
        # Find the index where timestamp exceeds end_time
        end_idx = np.searchsorted(ts, end_time)
        if end_idx > 0:  # Only include if we have data before end_time
            truncated_timestamps.append(ts[:end_idx])
            truncated_values.append(v[:end_idx])
    return truncated_timestamps, truncated_values

# Find minimum end times for both datasets
all_timestamps_5 = gumbel_timestamps_5 + mcts_timestamps_5 + ppo_timestamps_5
min_end_time_5 = min(ts[-1] for ts in all_timestamps_5)

all_timestamps_15 = gumbel_timestamps_15 + mcts_timestamps_15 + ppo_timestamps_15
min_end_time_15 = min(ts[-1] for ts in all_timestamps_15)

# Truncate all datasets to end at their respective minimum end times
gumbel_timestamps_5, gumbel_values_5 = truncate_to_time(gumbel_timestamps_5, gumbel_values_5, min_end_time_5)
mcts_timestamps_5, mcts_values_5 = truncate_to_time(mcts_timestamps_5, mcts_values_5, min_end_time_5)
ppo_timestamps_5, ppo_values_5 = truncate_to_time(ppo_timestamps_5, ppo_values_5, min_end_time_5)

gumbel_timestamps_15, gumbel_values_15 = truncate_to_time(gumbel_timestamps_15, gumbel_values_15, min_end_time_15)
mcts_timestamps_15, mcts_values_15 = truncate_to_time(mcts_timestamps_15, mcts_values_15, min_end_time_15)
ppo_timestamps_15, ppo_values_15 = truncate_to_time(ppo_timestamps_15, ppo_values_15, min_end_time_15)

# Function to process and plot data
def process_and_plot_data(timestamps, values, color, label, min_end_time):
    # Choose common timestamps covering all data
    min_time = 0  # Start from 0
    max_time = min_end_time  # Use the minimum end time we found
    common_time = np.linspace(min_time, max_time, 100)

    # Interpolate each dataset onto the common timestamps
    interp_values = []
    for t, v in zip(timestamps, values):
        interp_v = np.interp(common_time, t, v)
        interp_values.append(interp_v)

    interp_values = np.array(interp_values)

    # Compute mean, min, and max
    mean_values = np.mean(interp_values, axis=0)
    min_values = np.min(interp_values, axis=0)
    max_values = np.max(interp_values, axis=0)
    
    return common_time, mean_values, min_values, max_values

# Create figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Process and plot 5 blocks data
gumbel_time_5, gumbel_mean_5, gumbel_min_5, gumbel_max_5 = process_and_plot_data(gumbel_timestamps_5, gumbel_values_5, COLORS['gumbel'], 'Gumbel', min_end_time_5)
mcts_time_5, mcts_mean_5, mcts_min_5, mcts_max_5 = process_and_plot_data(mcts_timestamps_5, mcts_values_5, COLORS['mcts'], 'MCTS', min_end_time_5)
ppo_time_5, ppo_mean_5, ppo_min_5, ppo_max_5 = process_and_plot_data(ppo_timestamps_5, ppo_values_5, COLORS['ppo'], 'PPO', min_end_time_5)

# Plot 5 blocks data
ax1.plot(gumbel_time_5, gumbel_mean_5, color=COLORS['gumbel'], label='Gumbel', linewidth=2)
ax1.fill_between(gumbel_time_5, gumbel_min_5, gumbel_max_5, color=COLORS['gumbel'], alpha=0.1)
ax1.plot(mcts_time_5, mcts_mean_5, color=COLORS['mcts'], label='MCTS', linewidth=2)
ax1.fill_between(mcts_time_5, mcts_min_5, mcts_max_5, color=COLORS['mcts'], alpha=0.1)
ax1.plot(ppo_time_5, ppo_mean_5, color=COLORS['ppo'], label='PPO', linewidth=2)
ax1.fill_between(ppo_time_5, ppo_min_5, ppo_max_5, color=COLORS['ppo'], alpha=0.1)

# Add VTR line to first plot
ax1.axhline(y=2733, color=COLORS['vtr'], linestyle='--', linewidth=2, label='VTR opt')

ax1.set_xlabel('Wall Clock Time (s)', fontsize=12)
ax1.set_ylabel('HPWL', fontsize=12)
ax1.set_title('5 blocks n=50', fontsize=14, pad=15)
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.margins(x=0)

# Process and plot 15 blocks data
gumbel_time_15, gumbel_mean_15, gumbel_min_15, gumbel_max_15 = process_and_plot_data(gumbel_timestamps_15, gumbel_values_15, COLORS['gumbel'], 'Gumbel', min_end_time_15)
mcts_time_15, mcts_mean_15, mcts_min_15, mcts_max_15 = process_and_plot_data(mcts_timestamps_15, mcts_values_15, COLORS['mcts'], 'MCTS', min_end_time_15)
ppo_time_15, ppo_mean_15, ppo_min_15, ppo_max_15 = process_and_plot_data(ppo_timestamps_15, ppo_values_15, COLORS['ppo'], 'PPO', min_end_time_15)

# Plot 15 blocks data
ax2.plot(gumbel_time_15, gumbel_mean_15, color=COLORS['gumbel'], linewidth=2)
ax2.fill_between(gumbel_time_15, gumbel_min_15, gumbel_max_15, color=COLORS['gumbel'], alpha=0.1)
ax2.plot(mcts_time_15, mcts_mean_15, color=COLORS['mcts'], linewidth=2)
ax2.fill_between(mcts_time_15, mcts_min_15, mcts_max_15, color=COLORS['mcts'], alpha=0.1)
ax2.plot(ppo_time_15, ppo_mean_15, color=COLORS['ppo'], linewidth=2)
ax2.fill_between(ppo_time_15, ppo_min_15, ppo_max_15, color=COLORS['ppo'], alpha=0.1)

# Add VTR line to second plot
ax2.axhline(y=2733, color=COLORS['vtr'], linestyle='--', linewidth=2)

ax2.set_xlabel('Wall Clock Time (s)', fontsize=12)
ax2.set_title('15 blocks n=100', fontsize=14, pad=15)
ax2.grid(True, alpha=0.3)
ax2.margins(x=0)

# Set y-axis limits and ticks for both plots
# For 5 blocks plot
y_min_5 = min(min(gumbel_min_5), min(mcts_min_5), min(ppo_min_5))
y_max_5 = max(max(gumbel_max_5), max(mcts_max_5), max(ppo_max_5))
y_min_5 = y_min_5 - (y_max_5 - y_min_5) * 0.1  # Add 10% more space at the bottom
y_ticks_5 = np.linspace(y_min_5, y_max_5, 6)  # Create 6 evenly spaced ticks
y_ticks_5 = np.append(y_ticks_5, 2733)  # Add VTR value
y_ticks_5 = np.sort(y_ticks_5)  # Sort ticks
y_ticks_5 = np.round(y_ticks_5).astype(int)  # Convert to integers
ax1.set_ylim(y_min_5, y_max_5)
ax1.set_yticks(y_ticks_5)

# For 15 blocks plot
y_min_15 = min(min(gumbel_min_15), min(mcts_min_15), min(ppo_min_15))
y_max_15 = max(max(gumbel_max_15), max(mcts_max_15), max(ppo_max_15))
y_min_15 = y_min_15 - (y_max_15 - y_min_15) * 0.1  # Add 10% more space at the bottom
y_ticks_15 = np.linspace(y_min_15, y_max_15, 6)  # Create 6 evenly spaced ticks
y_ticks_15 = np.append(y_ticks_15, 2733)  # Add VTR value
y_ticks_15 = np.sort(y_ticks_15)  # Sort ticks
ax2.set_ylim(y_min_15, y_max_15)
ax2.set_yticks(y_ticks_15)

# Adjust layout to prevent overlap
plt.tight_layout()

plt.savefig('comparison_plot.png', dpi=300, bbox_inches='tight')
