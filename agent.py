import numpy as np


class Agent:
    def __init__(self, n_agents, state):
        """
        :param n_agents: number of agents (int)
        :param state: state object (state object, must be initialized using the State Class)
        """
        self.state = state.get_state()
        self.n_agents = n_agents
        self.grid_size = state.get_size()
        self.state_values = np.zeros((n_agents, self.grid_size))
        self.actions = np.chararray(self.state_values.shape)


    def value_iteratation(self, current_state):
        if 0 < current_state < (self.grid_size - 1):

            if sum(self.state_values[:, current_state - 1] + self.value_iteratation(current_state - 1)[:, current_state - 1])\
                >\
                sum(self.state_values[:, current_state + 1] + self.value_iteratation(current_state + 1)[:, current_state + 1])\
                    :

                self.state_values[:, current_state] = self.state_values[:, current_state - 1] + self.value_iteratation(current_state - 1)[:, current_state - 1]
                self.actions[:, current_state] = "left"
            else:
                self.state_values[:, current_state] = self.state_values[:, current_state + 1] + self.value_iteratation(current_state + 1)[:, current_state + 1]
                self.actions[:, current_state] = "right"

        return self.state_values

    def choose_action(self, current_state):
        actions = self.actions[:, current_state]
        return actions

    def get_state_values(self):
        return self.state_values

