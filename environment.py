import numpy as np


class State:
    """
    Creates an object to keep track of what is happening in the environment (agent positions)
    """
    def __init__(self, size, n_agents, init_state_distribution, rewards):
        """
        :param size: size (length) of the gridworld (int)
        :param n_agents: number of agents (int)
        :param init_position: starting position of all agents, counting from 0 (int)
        :param rewards: rewards for each agent (dict)
        """
        self.size = size
        self.init_state_distribution = init_state_distribution
        self.n_agents = n_agents
        self.rewards = rewards
        self.state = np.full((n_agents, 1), np.random.choice(size, p=init_state_distribution))

    def next_state(self, actions):
        """
        :param actions: list of actions for each agent in order 1,2,...,n for all agents (list)
        checks if all agent actions are the same and computes the next state
        :return: next global state (list)
        """

        next_state = self.state
        if all(elements == actions[0] for elements in actions):
            if actions[0] == "l":
                next_state = self.state - np.full(self.state.shape, 1)
            if actions[0] == "r":
                next_state = self.state + np.full(self.state.shape, 1)

            if 0 <= next_state[0] <= (self.size - 1):
                self.state = next_state

        return self.state.flatten().tolist()

    def reset(self):
        """
        resets global state to the initial state
        """
        self.state = np.full((self.n_agents, 1), np.random.choice(self.size, p=self.init_state_distribution))

    def give_reward(self):
        """
        :return: the rewards for each agent at the current state (list)
        """
        if self.state[0] == 0:
            return self.rewards["left"]
        if self.state[0] == self.size - 1:
            return self.rewards["right"]
        else:
            return [0 for i in range(self.n_agents)]

    def is_end(self):
        """
        checks if the agents are at the end of the grid and resets
        """
        if self.state[0] == 0 or self.state[0] == self.size - 1:
            self.reset()

    def show_grid(self):
        """
        prints the position of all agents in the grid
        """
        grid = np.zeros((self.n_agents, self.size))
        grid[:, self.state] = 1
        print(grid)

    def get_state(self):
        """
        :return: the current state of all agents (list)
        """
        return self.state.flatten().tolist()