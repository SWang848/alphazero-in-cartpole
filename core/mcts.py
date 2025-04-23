import math
from typing import Optional, Dict
import numpy as np
import ray
from copy import deepcopy

from config.base import BaseConfig
from core.util import MinMaxStats, sample_gumbel


@ray.remote
class GumbelSearchWorker:

    def __init__(
        self,
        config: BaseConfig,
    ):
        self.config = config
        self.env = config.env_creator(num_target_blocks=config.num_target_blocks)

    def process_batch(
        self,
        root_nodes,         # List of all root nodes
        task_assignments,   # List of (idx, batch_children) tuples to process
        mcts_windows,       # List of rolling windows for all environments
        min_max_stats,      # List of all min_max_stats
    ):

        # Initialize storage for results
        leaf_results = []  # Store tuples of (leaf_node, window, done, info, env_index)
        windows = deepcopy(mcts_windows)
        self.env.reset()
        best_found = {"hpwl": float("inf"), "reward": None, "state": None}
        
        # Process each assigned action
        for idx, batch_children in task_assignments:
            # Collect trajectories for this action across all environments
            trajectories = []
            for env_index, (root, child, window, stats) in enumerate(zip(
                root_nodes, batch_children, windows, min_max_stats
            )):
                node = root
                parent_q = 0
                trajectory = [node]
                
                # Start with the designated child for the root
                if node.expanded:
                    mean_q = node.mean_q(parent_q)
                    best_child = child  # Use the specified child
                    best_child.parent_traversed = node
                    
                    if best_child.expanded:
                        window.add(
                            best_child.obs,
                            best_child.env_state,
                            best_child.reward,
                            best_child.action,
                            best_child.info,
                        )
                    
                    node = best_child
                    trajectory.append(node)
                    
                    # Continue with normal traversal after the first step
                    while node.expanded:
                        mean_q = node.mean_q(parent_q)
                        best_child = node.best_child(stats, mean_q)
                        best_child.parent_traversed = node
                        
                        if best_child.expanded:
                            window.add(
                                best_child.obs,
                                best_child.env_state,
                                best_child.reward,
                                best_child.action,
                                best_child.info,
                            )
                        node = best_child
                        trajectory.append(node)
                
                trajectories.append((env_index, trajectory))
            
            # Process trajectories to perform environment steps
            for env_index, trajectory in trajectories:
                if len(trajectory) <= 1:
                    # Simulation ended at root or couldn't start
                    continue  # Skip this trajectory, no leaf node generated

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
                
                # Update window for the current environment
                current_window = windows[env_index]
                current_window.add(
                    obs["board_image"],
                    self.env.get_state(),
                    reward,
                    to_node.action,
                    info,
                )
                
                # Store results for this successful simulation step
                leaf_results.append((to_node, current_window, done, info, env_index))

        return leaf_results, best_found


class Node:
    def __init__(self, config, action, num_actions):
        self.config = config
        self.action = action
        self.num_actions = num_actions
        self.parent_traversed: Optional[Node] = None

        self.reward: Optional[float] = None
        self.obs: Optional[np.ndarray] = None
        self.env_state: Optional[any] = None
        self.info: Optional[dict] = None
        self.terminal: bool = False
        self.expanded: bool = False
        self.child_logits: Optional[np.ndarray] = None

        self.num_visits: int = 0
        self.value_sum: float = 0.0

        self.children: Dict[int, Node] = {}

        self.child_priors = None

    def expand(
        self,
        obs,
        reward,
        terminal,
        info,
        state,
        priors: np.ndarray,
        logits: np.ndarray = None,
    ):
        self.obs = obs
        self.reward = reward
        self.terminal = terminal
        self.env_state = state
        self.info = info
        self.child_logits = logits
        if terminal:
            return

        self.child_priors = priors
        for i in range(self.num_actions):
            self.children[i] = Node(self.config, i, self.num_actions)

        self.expanded = True

    def add_child_logits(self, logits):
        self.child_logits = logits

    def add_exploration_noise(self, noise, exploration_fraction):
        self.child_priors = np.where(
            self.info["action_mask"],
            self.child_priors * (1 - exploration_fraction)
            + noise * exploration_fraction,
            0.0,
        )
        # self.child_priors = np.where(self.child_priors != 0,
        #                              self.child_priors * (1 - exploration_fraction) + noise * exploration_fraction,
        #                              self.child_priors)

    def child_number_visits(self):
        return np.array([child.num_visits for _, child in self.children.items()])

    def child_values(self, min_max_stats, mean_q=None):
        values = []
        accu = max if self.config.max_reward_return else sum
        for _, child in self.children.items():
            child_value = child.mean_value()
            if child.num_visits > 0:
                child_value = min_max_stats.normalize(
                    accu([child.reward, self.config.gamma * child_value])
                )
            else:
                if mean_q is not None:
                    child_value = min_max_stats.normalize(mean_q)
                else:
                    child_value = 0.0
            # clip child_value to [0, 1] to avoid out of range
            child_value = min(max(child_value, 0.0), 1.0)
            values.append(child_value)
        return np.array(values)  # Return normalized values

    def mean_value(self):
        return self.value_sum / self.num_visits if self.num_visits > 0 else 0.0

    def mean_q(self, parent_q):
        # see EfficientZero p.18, 'mean_q' function
        total_q = 0
        total_visits = 0
        for _, child in self.children.items():
            if child.num_visits > 0:
                total_q += child.reward + self.config.gamma * child.mean_value()
                total_visits += 1

        if self.parent_traversed is None and total_visits > 0:
            mean_q = total_q / total_visits
        else:
            mean_q = (total_q + parent_q) / (total_visits + 1)

        return mean_q

    def get_child(self, action):
        return self.children[action]

    def puct_scores(self, min_max_stats, mean_q=None):
        # See: https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf
        # p. 17, Section "Search"
        c_base = self.config.c_base
        c_init = self.config.c_init
        c_term = np.log((1 + self.num_visits + c_base) / c_base) + c_init
        visit_term = np.sqrt(self.num_visits) / (self.child_number_visits() + 1)

        prior_score = c_term * visit_term * self.child_priors
        value_score = self.child_values(min_max_stats, mean_q)
        return value_score + prior_score

    def best_action(self, min_max_stats: MinMaxStats, mean_q):
        score = self.puct_scores(min_max_stats, mean_q)
        masked_score = np.where(self.info["action_mask"], score, -np.inf)
        # masked_score = np.where(self.child_priors != 0, score, -np.inf)
        max_val = np.max(masked_score)
        action = np.random.choice(np.argwhere(masked_score == max_val).flatten())
        return action

    def best_child(self, min_max_stats, mean_q):
        return self.children[self.best_action(min_max_stats, mean_q)]


class BatchTree:
    def __init__(self, root_num, num_actions, config):
        self.root_num = root_num
        self.action_shape = num_actions
        self.config = config

        self.roots = []
        for _ in range(root_num):
            root = Node(self.config, None, num_actions)
            self.roots.append(root)

        self.node_hash_tables = [{} for _ in range(root_num)]

    def prepare(self, mcts_windows, priors, logits):
        for i in range(self.root_num):
            prior = priors[i]
            state = mcts_windows[i].env_state
            root = self.roots[i]
            root.num_visits += 1
            info = mcts_windows[i].infos[0]
            logit = logits[i]
            if not root.expanded and not root.terminal:
                root.expand(mcts_windows[i].obs, None, False, info, state, prior, logit)
            elif not root.terminal and root.child_logits is None:
                root.add_child_logits(logit)

    def apply_actions(self, actions):
        for i in range(self.root_num):
            if actions[i] is None:
                continue

            root = self.roots[i]
            if root.terminal:
                continue

            action = actions[i]
            new_root = root.get_child(action)
            new_root.parent = None
            self.roots[i] = new_root

    def get_distributions(self):
        dists = []
        for root in self.roots:
            dists.append(root.child_number_visits())
        return dists

    def get_values(self):
        values = []
        for root in self.roots:
            values.append(root.mean_value())
        return values

    def get_children_values(self, min_max_stats, mean_q):
        values = []
        for root in self.roots:
            values.append(root.child_values(min_max_stats, mean_q))
        return values

    def clear(self):
        self.roots = None
        self.node_hash_tables = [{} for _ in range(self.root_num)]


class MCTS:
    def __init__(
        self,
        config,
        model,
    ):
        self.config = config
        self.model = model
        self._search_workers = []

        # Maximum number of parallel workers for Gumbel search
        self.max_parallel_searches = self.config.max_parallel_searches
        while len(self._search_workers) < self.max_parallel_searches:
            worker = GumbelSearchWorker.options(num_cpus=0.5).remote(self.config)
            self._search_workers.append(worker)

    def gumbel_squential_halving_search(self, roots, mcts_windows):
        # Ensure this instance has enough search workers
        while len(self._search_workers) < self.max_parallel_searches:
            worker = GumbelSearchWorker.options(num_cpus=0.5).remote(self.config)
            self._search_workers.append(worker)

        # Initialize best found for this specific search
        best_found = {"hpwl": float("inf"), "reward": None, "state": None}
        min_max_stats = [MinMaxStats() for _ in range(roots.root_num)]
        remaining_sim_budget = self.config.num_simulations
        m = self.config.m_top
        num_roots = roots.root_num

        # Get initial logits which already masked out invalid action's logits and apply Gumbel noise more efficiently
        batch_logits = np.stack([root.child_logits for root in roots.roots])
        gumbel_noise = sample_gumbel(batch_logits.shape)
        gumbel_logits = batch_logits + gumbel_noise

        # Select initial top m actions more efficiently
        top_m_indices = np.argsort(gumbel_logits, axis=1)[:, -m:][:, ::-1]
        selected_logits = np.take_along_axis(gumbel_logits, top_m_indices, axis=1)
        
        # Pre-compute child nodes for faster lookup
        roots_children = [root.children for root in roots.roots]
        selected_children = [[roots_children[i][action] for action in actions] 
                            for i, actions in enumerate(top_m_indices)]
        num_workers = len(self._search_workers)
        
        # Pre-allocate arrays for storing results
        all_leaf_nodes = []
        all_windows = []
        all_dones = []
        all_infos = []
        all_env_indices = []
        
        # Cache for child values - recalculated only when visits change
        children_value_cache = {}
        
        while remaining_sim_budget > 0:
            if m <= 3:
                num_sims_per_action = max(1, math.ceil(remaining_sim_budget / m))
            else:
                log2_m_top = np.log2(self.config.m_top) 
                num_sims_per_action = max(
                    1, math.floor(self.config.num_simulations / (m * log2_m_top))
                )
            
            # Pre-put shared objects in ray object store
            mcts_windows_ref = ray.put(mcts_windows)
            roots_ref = ray.put(roots.roots)
            min_max_stats_ref = ray.put(min_max_stats)
            
            for _ in range(num_sims_per_action):
                # More efficient batch processing
                worker_assignments = {}  # Map worker_idx -> list of tasks for this worker
                
                # Assign all actions across all workers
                for i in range(m):
                    batch_children = np.array(selected_children)[:, i]
                    worker_idx = i % num_workers
                    
                    if worker_idx not in worker_assignments:
                        worker_assignments[worker_idx] = []
                    
                    worker_assignments[worker_idx].append((i, batch_children))
                
                # Submit tasks to workers (one per worker with multiple actions to process)
                futures = []
                for worker_idx, tasks in worker_assignments.items():
                    # Each worker gets multiple actions to process in a single remote call
                    futures.append(
                        self._search_workers[worker_idx].process_batch.remote(
                            roots_ref,
                            tasks,  # List of (idx, batch_children) tuples
                            mcts_windows_ref,
                            min_max_stats_ref,
                        )
                    )
                
                # Process results
                batch_results = ray.get(futures)
                
                # Clear lists for current batch
                all_leaf_nodes.clear()
                all_windows.clear()
                all_dones.clear()
                all_infos.clear()
                all_env_indices.clear()
                
                # Process all worker results
                for worker_results in batch_results:
                    leaf_results, worker_best_found = worker_results
                    
                    # Update best found if better
                    if worker_best_found["hpwl"] < best_found["hpwl"]:
                        best_found = worker_best_found
                    
                    # Extend results
                    for leaf_node, window, done, info, env_index in leaf_results:
                        all_leaf_nodes.append(leaf_node)
                        all_windows.append(window)
                        all_dones.append(done)
                        all_infos.append(info)
                        all_env_indices.append(env_index)
                
                # Skip NN evaluation if no leaves to process
                if not all_leaf_nodes:
                    continue
                    
                # Perform batched NN inference
                priors, values, _ = self.model.compute_priors_and_values(all_windows)
                
                # Process each leaf
                for i in range(len(all_leaf_nodes)):
                    # Backpropagate results
                    self.backpropagate(
                        all_leaf_nodes[i],
                        all_windows[i],
                        values[i],
                        priors[i],
                        all_dones[i],
                        all_infos[i],
                        min_max_stats[all_env_indices[i]],
                    )
                
                # Invalidate cache after node visits change
                children_value_cache.clear()
            
            # Compute selection scores for remaining candidates
            # Calculate node visit counts (only needed once per batch)
            max_visits = np.array([
                max(child.num_visits for child in children)
                for children in selected_children
            ])
            
            # Calculate child values with caching
            children_values = np.zeros((num_roots, roots.roots[0].num_actions))
            for j in range(num_roots):
                if j not in children_value_cache:
                    children_value_cache[j] = roots.roots[j].child_values(min_max_stats[j], roots.roots[j].mean_q(0))
                children_values[j] = children_value_cache[j]
            
            # Calculate actions array
            actions = np.array([[child.action for child in children] 
                              for children in selected_children])
            
            # Compute transforms
            c_visit_term = (self.config.c_visit + max_visits[:, None]) * self.config.c_scale
            action_values = np.take_along_axis(children_values, actions, axis=1)
            sigma_transforms = c_visit_term * action_values
            sigma_logits = selected_logits + sigma_transforms
            
            # Update remaining budget and halve number of candidates
            remaining_sim_budget -= num_sims_per_action * m
            m = max(1, m // 2)

                
            # Reuse existing arrays for top indices
            top_m_indices = np.argsort(sigma_logits, axis=1)[:, -m:][:, ::-1]
            
            # Update selected children and logits
            new_selected_children = []
            for i, (children, indices) in enumerate(zip(selected_children, top_m_indices)):
                new_selected_children.append([children[j] for j in indices])
            selected_children = new_selected_children
            
            # Update selected logits
            selected_logits = np.take_along_axis(sigma_logits, top_m_indices, axis=1)
        
        # Get final selected actions and values
        selected_actions = [children[0].action for children in selected_children]
        
        # Calculate root values
        root_q_values = []
        for i in range(num_roots):
            if i in children_value_cache:
                root_q_values.append(children_value_cache[i])
            else:
                root_q_values.append(roots.roots[i].child_values(min_max_stats[i], roots.roots[i].mean_q(0)))
        
        return selected_actions, roots.get_values(), root_q_values, best_found

    def backpropagate(self, leaf_node, window, value, prior, done, info, min_max_stats):
        leaf_node.expand(
            window.latest_obs(), window.rewards[0], done, info, window.env_state, prior
        )

        accu = max if self.config.max_reward_return else sum
        if done:
            value = 0.0  # Terminal state value is 0

        current_node = leaf_node
        propagated_value = value
        while current_node is not None:
            current_node.value_sum += propagated_value
            current_node.num_visits += 1
            parent_node = current_node.parent_traversed

            if parent_node is not None:  # If not the root node
                reward = current_node.reward
                qsa = accu([reward, self.config.gamma * current_node.mean_value()])
                min_max_stats.update(qsa)
                propagated_value = accu([reward, self.config.gamma * propagated_value])

            else:  # This is the root node
                min_max_stats.update(current_node.mean_value())

            if parent_node is not None:
                current_node.parent_traversed = None
            current_node = parent_node
