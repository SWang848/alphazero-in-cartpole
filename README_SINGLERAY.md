# Single-Layer Ray Architecture

This document explains the single-layer Ray parallelization architecture implemented for the Gumbel Sequential Halving search in the AlphaZero algorithm.

## Overview

The original implementation used a nested Ray architecture:
1. Outer layer: `RolloutWorker` instances
2. Inner layer: `MCTS.gumbel_squential_halving_search` using `run_batch_simulation.remote()`

This nested approach led to several issues:
- Inefficient resource management 
- Complex debugging
- Potentially redundant Ray initialization
- Difficulty in controlling resource allocation

## New Architecture

The new single-layer architecture follows these principles:
1. Ray is initialized only once at the application level
2. MCTS no longer initializes Ray or creates dynamic Ray tasks
3. A pool of `GumbelSearchWorker` actors is maintained for parallel search

### Components

1. **GumbelSearchWorker**
   - Ray actor that processes a single action for multiple environments
   - Handles traversal, environment simulation, and data collection
   - Returns leaf nodes and updated windows to MCTSWorker for batch neural network evaluation

2. **MCTS**
   - Manages a pool of GumbelSearchWorker instances
   - Coordinates batch evaluation of leaf nodes
   - Performs backpropagation and Gumbel score updates

3. **MCTSWorker**
   - Manages overall MCTS search and environments
   - Creates MCTS instance which uses the shared GumbelSearchWorker pool

### Benefits

1. **Clearer Resource Management**
   - Workers are created once and reused
   - No dynamic creation of Ray resources during search

2. **Simplified Debugging**
   - Clearer separation of responsibilities
   - Errors can be traced to specific GumbelSearchWorker instances

3. **Better Batch Efficiency**
   - Leaf nodes from all actions can be evaluated in a single batch
   - Reduces neural network inference overhead

4. **Configurable Parallelism**
   - Adjust `max_parallel_searches` to control parallelism level
   - Scales based on available resources

## Configuration

Add to your config:

```python
config = Config(
    # Other parameters...
    max_parallel_searches=8,  # Number of parallel search workers
)
```

## Implementation Notes

1. The `GumbelSearchWorker` class in `core/workers.py` handles simulation for a batch of environments for a single action
2. Each worker needs minimal resources (typically 1 CPU)
3. Neural network inference is still performed by the main MCTS instance, leveraging batch evaluation
4. Workers are shared across all MCTS instances to maximize resource utilization

## Scaling Considerations

- For small search spaces, use fewer workers (4-8)
- For large search spaces with many actions, increase `max_parallel_searches` (16-32)
- CPU-intensive environments may require more CPUs per worker
- Balance between number of RolloutWorkers and GumbelSearchWorkers based on your available resources 