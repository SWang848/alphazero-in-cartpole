from copy import deepcopy
import random
import math
import numpy as np

from core.util import MinMaxStats


class Node:
    def __init__(self, config, action, num_actions):
        self.config = config
        self.action = action
        self.num_actions = num_actions
        self.parent_traversed = None

        self.reward = None
        self.env_state = None
        self.info = None
        self.terminal = False
        self.expanded = False

        self.num_visits = 0
        self.value_sum = 0

        self.children = {}

    def expand(self, reward, terminal, info, state):
        self.reward = reward
        self.terminal = terminal
        self.env_state = state
        self.info = info

        if terminal:
            return

        for i in range(self.num_actions):
            self.children[i] = Node(self.config, i, self.num_actions)

        self.expanded = True

    def child_number_visits(self):
        return np.array([child.num_visits for _, child in self.children.items()])

    def child_values(self, min_max_stats, mean_q=None):
        values = []
        accu = max if self.config.max_reward_return else sum
        for _, child in self.children.items():  # Update min-max stats
            child_value = child.mean_value()
            if child.num_visits > 0:
                min_max_stats.update(
                    accu([child.reward, self.config.gamma * child_value])
                )
            # if child is not visited, update min-max stats with mean_q
            else:
                if mean_q is not None:
                    min_max_stats.update(mean_q)

        for _, child in self.children.items():  # Calculate child values
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

    def ucb_scores(self, min_max_stats, mean_q=None):
        c_base = self.confg.c_base
        
        value_score = self.child_values(min_max_stats, mean_q)
        ucb_scores = value_score + c_base * np.sqrt(np.log(self.num_visits) / (self.child_number_visits() + 1e-8))
        return ucb_scores

    def best_action(self, min_max_stats: MinMaxStats, mean_q):
        score = self.ucb_scores(min_max_stats, mean_q)
        masked_score = np.where(self.info["action_mask"], score, -np.inf)
        max_val = np.max(masked_score)
        action = np.random.choice(np.argwhere(masked_score == max_val).flatten())
        return action

    def choose_child(self, min_max_stats, mean_q, **kwargs):
        # return self.children[self.best_action(min_max_stats, mean_q)]
        score = self.ucb_scores(min_max_stats, mean_q)
        masked_score = np.where(self.info["action_mask"], score, -np.inf)
        sorted_desc_score = np.argsort(masked_score)[::-1]
        sorted_desc_score = sorted_desc_score[
            : np.sum(np.isfinite(masked_score))
        ]  # remove invalid actions from the sorted list
        
        best_action = sorted_desc_score[0]
        best_child = self.children[best_action]
        if kwargs["forced_exploration"]:
            if best_child.num_visits < kwargs["threshold_n"]:
                return best_child
            else:
                top_actions = sorted_desc_score[: kwargs["num_top_actions"]]
                top_children = [self.children[action] for action in top_actions]
                eligible_children = [
                    child for child in top_children if child.num_visits < kwargs["threshold_n"]
                ]
                return random.choice(eligible_children)
        else:
            return best_child


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

    def prepare(self, mcts_windows):
        for i in range(self.root_num):
            state = mcts_windows[i].env_state
            root = self.roots[i]
            root.num_visits += 1
            info = mcts_windows[i].infos[0]

            if not root.expanded and not root.terminal:
                root.expand(mcts_windows[i].obs, None, False, info, state)

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

    def traverse(self, mcts_windows, min_max_stats):
        trajectories = []
        for i in range(self.root_num):
            node = self.roots[i]
            parent_q = 0

            trajectories.append([node])

            while node.expanded:
                mean_q = node.mean_q(parent_q)
                if (
                    node == self.roots[i] and self.config.forced_exploration
                ):  # the forced exploration only works on the roots kids' layer.
                    threshold_n = math.ceil(
                        self.config.k
                        * self.config.num_simulations
                        / np.sum(node.info["action_mask"])
                    )
                    num_top_actions = math.ceil(
                        self.config.percentage * np.sum(node.info["action_mask"])
                    )
                    
                    if sum(1 for child in node.children.values() if child.num_visits >= threshold_n) >= num_top_actions:  
                        # the forced exlpration only works if the top range of nodes are rarely visited.
                        forced_exploration = False
                    else:
                        forced_exploration = True
                    best_child = node.choose_child(
                        min_max_stats[i], mean_q, forced_exploration=forced_exploration, threshold_n=threshold_n, num_top_actions=num_top_actions
                    )
                else:
                    forced_exploration = False
                    best_child = node.choose_child(
                        min_max_stats[i], mean_q, forced_exploration=forced_exploration
                    )
                best_child.parent_traversed = node
                if (
                    best_child.expanded
                ):  # We can not do node.obs at the beginning of the loop, because the root node is already inside the sliding window
                    mcts_windows[i].add(
                        best_child.obs,
                        best_child.env_state,
                        best_child.reward,
                        best_child.action,
                        best_child.info,
                    )
                node = best_child
                trajectories[-1].append(node)
        return trajectories

    def backpropagate(
        self, leaf_nodes, mcts_windows, values, terminals, infos, min_max_stats
    ):
        for i in range(len(leaf_nodes)):
            node: Node = leaf_nodes[i]  # Take vals for current leaf_node
            reward = mcts_windows[i].rewards[0]
            state = mcts_windows[i].env_state
            value = values[i]
            terminal = terminals[i]
            info = infos[i]

            # Expand the leaf node
            # If it's a terminal node, the `expand` call will return without expansion
            node.expand(reward, terminal, info, state)

            # Check if node already exists
            # If yes, take it from the hash table
            if self.config.hash_nodes:
                node_hash = self.config.hash_env_state(node.env_state)
                if node_hash in self.node_hash_tables[i]:
                    node_ = self.node_hash_tables[i][
                        node_hash
                    ]  # Get the node from the hash table
                    node_.parent_traversed = (
                        node.parent_traversed
                    )  # Set attribute for backpropagation
                    node_.parent_traversed.children[node.action] = (
                        node_  # Set the child of the parent node to the preexisting node
                    )
                    node = node_
                else:
                    self.node_hash_tables[i][node_hash] = node

            # Define return accumulation function. For vanilla RL, we use R_t = r_t + gamma R_(t+1)
            # If `config.max_reward_return`, R_t = max(r_t, R_(t_1))
            accu = max if self.config.max_reward_return else sum
            if terminal:
                value = 0.0
            while True:
                node.value_sum += value
                # node.value_sum = (node.num_visits * node.value_sum + value) / (node.num_visits + 1)
                node.num_visits += 1
                min_max_stats[i].update(node.mean_value())

                if node.parent_traversed is None:
                    break
                reward = node.reward
                value = accu([reward, self.config.gamma * value])
                parent_node = node.parent_traversed
                node.parent_traversed = None
                node = parent_node

    def clear(self):
        self.roots = None
        self.node_hash_tables = [{} for _ in range(self.root_num)]


class MCTS:
    def __init__(
        self,
        config,
    ):
        self.config = config
        self.env = config.env_creator(num_target_blocks=config.num_target_blocks)

    def search(self, roots, mcts_windows):
        # Do one step of Batch MCTS
        min_max_stats = [MinMaxStats() for _ in range(roots.root_num)]
        self.env.reset()
        best_found = {"hpwl": float("inf"), "reward": None, "state": None}
        
        for simulation_index in range(self.config.num_simulations):
            windows = deepcopy(mcts_windows)
            trajectories = roots.traverse(windows, min_max_stats)

            dones = []
            leaf_nodes = []
            infos = []
            values = []

            for env_index in range(roots.root_num):
                trajectory = trajectories[env_index]
                if len(trajectory) == 1:
                    dones.append(True)
                    continue

                from_node = trajectory[-2]
                to_node = trajectory[-1]

                self.env = self.env.set_state(from_node.env_state)
                obs, reward, done, truncated, info = self.env.step(to_node.action)

                windows[env_index].add(
                    obs["board_image"],
                    self.env.get_state(),
                    reward,
                    to_node.action,
                    info,
                )
                leaf_nodes.append(to_node)
                dones.append(done)
                infos.append(info)
                values.append(self.random_rollout(self.env, done))
                if (
                    info["hpwl"] < best_found["hpwl"]
                ):  # record the best placement during the search.
                    best_found["hpwl"] = info["hpwl"]
                    best_found["reward"] = reward
                    best_found["state"] = self.env.get_state()   
            debug = self.config.debug
            plot = True

            roots.backpropagate(
                leaf_nodes, windows, values, dones, infos, min_max_stats
            )
            if debug and plot:
                from core.util import plot_tree
                import os

                root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                index = roots.roots[0].info["episode_steps"]
                # plotting the tree when the agent finishes the simulation in the roots layer.
                if simulation_index == self.config.num_simulations - 1:
                    plot_tree(
                        roots.roots[0],
                        leaf_nodes[0],
                        values[0],
                        min_max_stats[0],
                        output_file=os.path.join(
                            root_path,
                            f"evaluation/{os.path.basename(self.config.model_dir)}/{os.path.basename(self.config.model_path)}/tree_{index}.gv",
                        ),
                    )

        return roots.get_distributions(), roots.get_values(), best_found

    def random_rollout(self, env, done):
        if done:
            return 0.0
        
        total_reward = 0
        gamma = self.config.gamma
        step = 0
        
        env_state = env.get_state()  # Save initial state
        
        while not done:
            action_mask = env.get_mask()
            valid_actions = np.where(action_mask)[0]
            action = np.random.choice(valid_actions)
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += (gamma ** step) * reward  # Apply discount correctly
            step += 1
        
        env.set_state(env_state)  # Restore initial state
        return total_reward