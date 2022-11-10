import numpy as np


class Agent(object):
    """
    Creates an object for an agent
    """
    def __init__(self, id, n_states, n_agents, actions):
        """
        :param id: id number for the agent (int)
        :param n_states: number of states in the environment (int)
        :param n_agents: number of agents in the environment (int)
        :param actions: possible actions for the agent (tuple)
        """
        self.id = id
        self.actions = actions

        self.policy_distribution = np.full((len(actions), n_agents, n_states), 1 / len(actions))
        self.actions_select = np.empty((len(actions), n_agents, n_states), dtype=str)
        for i in range(len(actions)):
            self.actions_select[i, :, :] = actions[i]

        self.q_values = np.zeros((len(actions), n_states))

    def choose_action(self, state):
        """
        :param state: the state of all agents (list)
        :return: actions of all agents (list)
        """
        actions = []
        for i in range(len(state)):
            actions.append(np.random.choice(self.actions_select[:, i, state[i]].flatten(),
                                            p=self.policy_distribution[:, i, state[i]].flatten()))

        return actions
