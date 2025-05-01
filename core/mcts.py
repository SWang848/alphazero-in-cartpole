import math
from typing import Optional, Dict
import numpy as np
import ray
from collections import defaultdict

from core.util import MinMaxStats, sample_gumbel
from core.search_worker import GumbelSearchWorker

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
            
    def prepare_subtrees(self, m, selected_children, mcts_windows):
        """
        Prepares subtrees across multiple workers in parallel.
        
        Args:
            m: Number of top actions to consider
            selected_children: List of selected child nodes for each root
            mcts_windows: Environment windows for observation
            
        Returns:
            tuple: (root_nodes, windows) where:
                  - root_nodes: List of lists of action nodes with shape [num_env, num_node]
                  - windows: List of lists of observation windows with shape [num_env, num_node]
        """
        num_workers = len(self._search_workers)
        prepare_tasks = defaultdict(list)
        mcts_windows_ref = ray.put(mcts_windows)
        
        batch_children = np.array(selected_children)
        for i in range(m):
            worker_idx = i % num_workers
            prepare_tasks[worker_idx].append((batch_children[:, i]))
        
        futures = [
            self._search_workers[worker_idx].initialize_subtree_roots.remote(
                mcts_windows_ref,
                tasks,
                worker_idx
            )
            for worker_idx, tasks in prepare_tasks.items()
        ]

        results = ray.get(futures)
        num_envs = len(mcts_windows)
        
        root_nodes = [[] for _ in range(num_envs)]
        windows = [[] for _ in range(num_envs)]
        worker_node_window_map = defaultdict(list)

        for window_results, worker_idx in results:
            for window, action_node, env_idx in window_results:
                root_nodes[env_idx].append(action_node)
                windows[env_idx].append(window)
                worker_node_window_map[worker_idx].append((action_node, window))
                
        all_windows = [window for env_windows in windows for window in env_windows]
        priors, _, _ = self.model.compute_priors_and_values(all_windows)
        
        expand_tasks = defaultdict(list)
        i = 0
        for worker_idx, node_window_pairs in worker_node_window_map.items():
            for action_node, window in node_window_pairs:
                expand_tasks[worker_idx].append((action_node, window, priors[i]))
                i += 1
        
        ray.get([
            self._search_workers[worker_idx].expand_subtrees.remote(expand_tasks)
            for worker_idx, tasks in expand_tasks.items()
        ])
        
        return root_nodes, windows

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
        roots_children = [root.children for root in roots.roots]
        selected_children = [[roots_children[i][action] for action in actions] 
                            for i, actions in enumerate(top_m_indices)]
        
        # Prepare subtrees across workers
        root_nodes, windows = self.prepare_subtrees(m, selected_children, mcts_windows)
            
        # Pre-allocate arrays for storing results
        all_leaf_nodes = []
        all_windows = []
        all_dones = []
        all_infos = []
        all_env_indices = []
        all_worker_indices = []
        
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
            
            for _ in range(num_sims_per_action):

                worker_assignments = defaultdict(list)
                min_max_stats_ref = ray.put(min_max_stats)
                worker_idx = 0
                num_workers = len(self._search_workers)
                
                # Group subtree roots by worker for balanced distribution
                for env_idx in range(len(root_nodes)):
                    for node_idx, (root_node, window) in enumerate(zip(root_nodes[env_idx], windows[env_idx])):
                        if node_idx < m:  # Only process top m actions
                            curr_worker = worker_idx % num_workers
                            worker_assignments[curr_worker].append((env_idx, root_node, window))
                            worker_idx += 1
                
                futures = []
                for worker_idx, tasks in worker_assignments.items():
                    futures.append(
                        self._search_workers[worker_idx].traverse.remote(
                            tasks,  # List of (env_idx, root_node, window) tuples
                            min_max_stats_ref,
                            worker_idx,
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
                all_worker_indices.clear()
                
                # Process all worker results
                for worker_results in batch_results:
                    leaf_results, worker_best_found = worker_results
                    
                    # Update best found if better
                    if worker_best_found["hpwl"] < best_found["hpwl"]:
                        best_found = worker_best_found
                    
                    for leaf_node, window, done, info, env_index, worker_idx in leaf_results:
                        all_leaf_nodes.append(leaf_node)
                        all_windows.append(window)
                        all_dones.append(done)
                        all_infos.append(info)
                        all_env_indices.append(env_index)
                        all_worker_indices.append(worker_idx)
                    
                # Perform batched NN inference
                priors, values, _ = self.model.compute_priors_and_values(all_windows)
                
                # Organize backpropagation tasks by worker
                backprop_tasks = defaultdict(list)
                
                # Group backpropagation tasks by worker to minimize RPC calls
                for i in range(len(all_leaf_nodes)):
                    worker_idx = all_worker_indices[i]
                    backprop_tasks[worker_idx].append((
                        all_env_indices[i],
                        all_leaf_nodes[i],
                        all_windows[i],
                        values[i],
                        priors[i],
                        all_dones[i],
                        all_infos[i]
                    ))
                
                # Dispatch backpropagation to workers in parallel
                backprop_results = ray.get([
                    self._search_workers[worker_idx].backpropagate.remote(
                        tasks,
                        min_max_stats_ref
                    )
                    for worker_idx, tasks in backprop_tasks.items()
                ])
                
                root_nodes = [[] for _ in range(num_roots)]
                for worker_root_nodes, _ in backprop_results:
                    for env_idx, nodes in worker_root_nodes.items():
                        root_nodes[env_idx].extend(nodes)
                
                for env_idx in range(len(min_max_stats)):
                    all_maximums = [max_vals[env_idx] for _, (_, max_vals) in backprop_results]
                    all_minimums = [min_vals[env_idx] for _, (min_vals, _) in backprop_results]
                    
                    # Update with the maximum of maximums and minimum of minimums
                    if all_maximums:  # Check if list is not empty
                        min_max_stats[env_idx].maximum = max(min_max_stats[env_idx].maximum, *all_maximums)
                    if all_minimums:  # Check if list is not empty
                        min_max_stats[env_idx].minimum = min(min_max_stats[env_idx].minimum, *all_minimums)
                
                # Invalidate cache after node visits change
                children_value_cache.clear()
            
            # Calculate node visit counts (only needed once per batch)
            max_visits = np.array([
                max(node.num_visits for node in env_nodes)
                for env_nodes in root_nodes
            ])
            
            
            # Select logits and calculate action values in a single pass
            selected_logits = np.zeros((num_roots, m))
            action_values = np.zeros((num_roots, m))
            for env_idx, env_nodes in enumerate(root_nodes):
                for node_idx, node in enumerate(env_nodes):
                    selected_logits[env_idx, node_idx] = gumbel_logits[env_idx, node.action]
                    reward = node.reward if node.reward is not None else 0
                    action_values[env_idx, node_idx] = reward + self.config.gamma * node.mean_value()
            # Compute the Gumbel sigma transformation
            c_visit_term = (self.config.c_visit + max_visits[:, None]) * self.config.c_scale
            sigma_transforms = c_visit_term * action_values
            
            # Add transformation to selected logits
            sigma_logits = selected_logits + sigma_transforms
            
            # Update remaining budget and halve number of candidates
            remaining_sim_budget -= num_sims_per_action * m
            m = max(1, m // 2)
 
            # Reuse existing arrays for top indices
            top_m_indices = np.argsort(sigma_logits, axis=1)[:, -m:][:, ::-1]
            
            # Reshape root_nodes based on top_m_indices
            # Use list comprehension and numpy indexing for better performance
            root_nodes = [
                [nodes[idx] for idx in indices]
                for nodes, indices in zip(root_nodes, top_m_indices)
            ]
            
            # Update windows to match root_nodes (if needed for future iterations)
            windows = [
                [windows[env_idx][idx] for idx in indices ]
                for env_idx, indices in enumerate(top_m_indices)
            ]
            
            # Update selected logits based on top indices
            selected_logits = np.take_along_axis(sigma_logits, top_m_indices, axis=1)
            
        
        # Get final selected actions and values
        selected_actions = [nodes[0].action if nodes else None for nodes in root_nodes]
        
        # Calculate root values
        root_q_values = []
        for i in range(num_roots):
            if i in children_value_cache:
                root_q_values.append(children_value_cache[i])
            else:
                root_q_values.append(roots.roots[i].child_values(min_max_stats[i], roots.roots[i].mean_q(0)))
        
        return selected_actions, roots.get_values(), root_q_values, best_found