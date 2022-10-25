import numpy as np


class State:
    """
    Creates an object to keep track of what is happening in the environment (agents positions)
    """
    def __init__(self, size, n_agents, init_position):
        """
        :param size: size (length) of the gridworld (int)
        :param n_agents: number of agents (int)
        :param init_position: starting position of all agents, counting from 0 (int)
        """
        self.size = size
        self.init_position = init_position
        self.n_agents = n_agents
        self.state = np.full((n_agents, 1), init_position)

    def next_state(self, actions):
        """
        :param actions: list of actions for each agent in order 1,2,...,n for all agents (list)
        all actions must be identical or all agents will do nothing
        :return: next global state (np array)
        """
        if all(elements == actions[0] for elements in actions):
            if actions[0] == "left":
                next_state = self.state - np.full(self.state.shape, 1)
            if actions[0] == "right":
                next_state = self.state + np.full(self.state.shape, 1)

            if 0 <= next_state[0] <= (self.size - 1):
                self.state = next_state
                return self.state

        return self.state

    def reset(self):
        """
        resets global state to the initial state
        """
        self.state = np.full((self.n_agents, 1), self.init_position)

    def show_grid(self):
        """
        prints the position of all agents in the grid
        """
        grid = np.zeros((self.n_agents, self.size))
        grid[:, self.state] = 1
        print(grid)
