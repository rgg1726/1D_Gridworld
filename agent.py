import numpy as np
from scipy.special import softmax


class Agent(object):
    """
    Creates an object for an agent
    """
    def __init__(self, id, n_states, n_agents):
        """
        :param id: id number for the agent (int)
        :param n_states: number of states in the environment (int)
        :param n_agents: number of agents in the environment (int)
        """
        self.id = id
        self.n_states = n_states
        self.n_agents = n_agents
        self.actions = (-1, 1)  # possible actions -1 for left, 1 for right
        self.pi = np.full((len(self.actions), n_agents, n_states), 1 / len(self.actions))   # initial policy distribution
        self.action_sampling = np.zeros((len(self.actions), n_agents, n_states), dtype=np.int64)
        for i in range(len(self.actions)):
            self.action_sampling[i, :, :] = self.actions[i] # fill the sampling matrix with actions to be picked by pi
        self.q_values = np.zeros((len(self.actions), n_agents, n_states))   # initial global q values

    def choose_action(self, state):
        """
        :param state: the state of all agents (np array)
        :return: actions of all agents (np array)
        """
        actions = np.zeros(state.shape, int)
        for i in range(state.shape[0]):
            actions[i] = np.random.choice(self.action_sampling[:, i, state[i]].flatten(),
                                          p=self.pi[:, i, state[i]].flatten())

        return actions

    def policy_update(self, q_i, alpha):
        """
        policy improvement step using optimized q values
        :param q_i: optimized q_values
        :param alpha: learning rate alpha
        """
        self.q_values = q_i

        for i in range(self.n_agents):
            self.pi[:, i, np.arange(self.n_states)] = softmax(alpha * q_i[:, i, np.arange(self.n_states)] +
                                                              np.log(self.pi[:, i, np.arange(self.n_states)] + 1e-16),
                                                              axis=0)
