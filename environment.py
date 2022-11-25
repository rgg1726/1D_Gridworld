import numpy as np


class State:
    """
    Creates an object to keep track of what is happening in the environment (agent positions)
    """
    def __init__(self, n_states, n_agents, init_state_distribution, rewards):
        """
        :param n_states: size (length) of the gridworld (int)
        :param n_agents: number of agents (int)
        :param init_state_distribution: initial state distribution of agents (np array)
        :param rewards: rewards for each agent (dict)
        """
        self.n_states = n_states
        self.init_state_distribution = init_state_distribution
        self.n_agents = n_agents
        self.rewards = rewards
        self.state = np.full((n_agents, 1), np.random.choice(n_states, p=init_state_distribution), dtype=np.int64)

    def next_state(self, actions):
        """
        :param actions: actions for each agent (-1, +1) (np array)
        checks if all agent actions are the same and computes the next state
        :return: next global state (np array)
        """
        if np.all(actions == actions[0]):
            self.state = self.state + actions

        return self.state

    def reset(self):
        """
        resets global state to the initial state
        """
        self.state = np.full((self.n_agents, 1), np.random.choice(self.n_states, p=self.init_state_distribution), dtype=np.int64)

    def give_reward(self):
        """
        :return: the rewards for each agent at the current state (list)
        """
        if self.state[0] == 0:
            return self.rewards["left"]
        if self.state[0] == self.n_states - 1:
            return self.rewards["right"]
        else:
            return [0 for _ in range(self.n_agents)]

    def is_end(self):
        """
        checks if the agents are at the end of the grid and resets
        """
        if self.state[0] == 0 or self.state[0] == self.n_states - 1:
            self.reset()

    def show_grid(self):
        """
        prints the position of all agents in the grid
        """
        grid = np.zeros((self.n_agents, self.n_states))
        grid[:, self.state] = 1
        print(grid)

    def get_state(self):
        """
        :return: the current state of all agents (np array)
        """
        return self.state
