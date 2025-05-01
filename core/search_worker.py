from copy import deepcopy
from collections import defaultdict

import ray

from config.base import BaseConfig


@ray.remote
class GumbelSearchWorker:

    def __init__(
        self,
        config: BaseConfig,
    ):
        self.config = config
        self.env = config.env_creator(num_target_blocks=config.num_target_blocks)
    
    def initialize_subtree_roots(
        self,
        mcts_windows,
        tasks,
        worker_idx,
    ):
        """
        Initialize subroots for given action nodes.
        
        Args:
            mcts_windows: List of (env_idx,) windows
            tasks: List of (batch_nodes) pairs to process
            worker_idx: Index of this worker
            
        Returns:
            tuple: (results, worker_idx) where results contains (window, action_nodes, task_idx, env_idx) tuples
        """
        self.env.reset()

        results = []
        
        for batch_nodes in tasks:
            windows = deepcopy(mcts_windows)
            for env_idx, (action_node, window) in enumerate(zip(batch_nodes, windows)):
                self.env.set_state(window.env_state)
                obs, reward, done, truncated, info = self.env.step(action_node.action)
                
                window.add(
                    obs["board_image"],
                    self.env.get_state(),
                    reward,
                    action_node.action,
                    info,
                    done
                )
                
                results.append((window, action_node, env_idx))
                
        return results, worker_idx

    def expand_subtree_roots(
        self,
        node_windows_priors,
    ):
        """
        Apply neural network priors to expand the initialized subtree roots.
        
        Args:
            node_windows_priors: List of (action_node, window, prior) tuples
        """
        for action_node, window, prior in node_windows_priors:
            action_node.expand(
                window.latest_obs(),
                None,
                window.dones[0],  
                window.infos[0],
                window.env_state,
                prior,
            )

    def traverse(
        self,
        tasks,          
        min_max_stats, 
        worker_idx, 
    ):
        """
        Traverse the search tree from given root nodes.
        
        Args:
            tasks: List of (env_idx, action_node, window) tuples to process
            min_max_stats: Statistics for value normalization
            worker_idx: Worker index
            
        Returns:
            tuple: (leaf_results, best_found) with leaf nodes and best solution found
        """
        # Initialize storage for results
        leaf_results = []  # Store tuples of (leaf_node, window, done, info, env_index)
        best_found = {"hpwl": float("inf"), "reward": None, "state": None}
        
        # Process each assigned task
        for env_idx, action_node, window in tasks:
            # Start from the action node
            node = action_node
            parent_q = 0
            trajectory = [node]
            
            # Traverse until reaching a leaf
            while node.expanded:
                mean_q = node.mean_q(parent_q)
                best_child = node.best_child(min_max_stats[env_idx], mean_q)
                
                best_child.parent_traversed = node
                
                if best_child.expanded:
                    window.add(
                        best_child.obs,
                        best_child.env_state,
                        best_child.reward,
                        best_child.action,
                        best_child.info,
                        best_child.terminal
                    )
                node = best_child
                trajectory.append(node)
            
            # Skip if trajectory is too short
            if len(trajectory) <= 1:
                continue
                
            # Get the last two nodes in the trajectory
            from_node = trajectory[-2]
            to_node = trajectory[-1]
            
            # Take environment step
            self.env.set_state(from_node.env_state)
            obs, reward, done, truncated, info = self.env.step(to_node.action)
            
            # Update local best found solution if applicable
            if "hpwl" in info and info["hpwl"] < best_found["hpwl"]:
                best_found = {
                    "hpwl": info["hpwl"],
                    "reward": reward,
                    "state": self.env.get_state(),
                }
            
            # Update window with new observation
            window.add(
                obs["board_image"],
                self.env.get_state(),
                reward,
                to_node.action,
                info,
                done
            )
            
            # Store results for this successful simulation step
            leaf_results.append((to_node, window, done, info, env_idx, worker_idx))
                
        return leaf_results, best_found

    def backpropagate(
        self, 
        tasks,
        min_max_stats
        ):
        """
        Backpropagate values from leaf nodes to roots.
        
        Args:
            tasks: List of (env_idx, leaf_node, window, value, prior, done, info) tuples
            min_max_stats: Statistics for value normalization
            
        Returns:
            tuple: (root_nodes_by_env, min_max_stats) where:
                 - root_nodes_by_env: Dictionary mapping env_idx to list of root nodes
                 - min_max_stats: Updated min_max_stats
        """
        # Track min/max values per environment
        min_values = [float('inf')] * len(min_max_stats)
        max_values = [float('-inf')] * len(min_max_stats)
        
        # Create dictionary to collect nodes by environment
        root_nodes = defaultdict(list)
        
        for env_idx, leaf_node, window, value, prior, done, info in tasks:
            leaf_node.expand(
                window.latest_obs(), window.rewards[0], done, info, window.env_state, prior
            )

            accu = max if self.config.max_reward_return else sum
            if done:
                value = 0.0  # Terminal state value is 0

            current_node = leaf_node
            propagated_value = value
            while True:
                current_node.value_sum += propagated_value
                current_node.num_visits += 1
                if current_node.parent_traversed is None:
                    break
                reward = current_node.reward
                qsa = accu([reward, self.config.gamma * current_node.mean_value()])
                min_max_stats[env_idx].update(qsa)
                
                # Track min/max values for this environment
                min_values[env_idx] = min(min_values[env_idx], min_max_stats[env_idx].minimum)
                max_values[env_idx] = max(max_values[env_idx], min_max_stats[env_idx].maximum)
                
                propagated_value = accu([reward, self.config.gamma * propagated_value])
                parent_node = current_node.parent_traversed
                current_node.parent_traversed = None
                current_node = parent_node

            # Add the root node to the appropriate environment group
            root_nodes[env_idx].append(current_node)
            
        # Return the reshaped root nodes and min/max stats values
        return root_nodes, (min_values, max_values)
    