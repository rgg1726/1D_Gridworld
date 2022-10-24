import numpy as np


class State:
    def __init__(self, size, n_agents, init_position, left_rewards, right_rewards):
        """
        :param size: size (length) of the gridworld (int)
        :param n_agents: number of agents (int)
        :param init_position: starting position of all agents (int)
        :param left_rewards: agent rewards for reaching the left edge (array)
        :param right_rewards: agent rewards for reaching the right edge (array)
        """
        self.size = size
        self.grid = np.zeros(size)
        self.n_agents = n_agents
        self.state = np.full((n_agents, 1), init_position)
        self.left_rewards = left_rewards
        self.right_rewards = right_rewards
        self.is_end = False

    def compute_reward(self):
        if np.array_equal(self.state, np.zeros(self.state.shape)):
            return self.left_rewards
        elif np.array_equal(self.state, np.full(self.state.shape, self.size-1)):
            return self.right_rewards
        return 0

    def next_state(self, actions):
        next_state = self.state
        if np.all(actions == actions[0]):
            if actions[0] == "left":
                next_state = self.state - np.full(actions.shape, 1)
            if actions[0] == "right":
                next_state = self.state + np.full(actions.shape, 1)

            if 0 <= next_state[0] <= (self.size - 1):
                return next_state
            return self.state
        return self.state

    def is_end(self):
        if (self.state[0] == 0) or (self.state[0] == self.size - 1):
            self.is_end = True

    def get_state(self):
        return self.state

    def get_size(self):
        return self.size
