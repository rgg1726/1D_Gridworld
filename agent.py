import numpy as np


class Agent(object):
    def __init__(self, id, n_states, n_agents, actions):
        self.id = id
        self.actions = actions

        self.policy_distribution = np.full((len(actions), n_agents, n_states), 1 / len(actions))
        self.selection_matrix = np.empty((len(actions), n_agents, n_states), dtype=str)
        for i in range(len(actions)):
            self.selection_matrix[i, :, :] = actions[i]

        self.q_values = np.zeros((len(actions), n_states))

    def choose_action(self, state):
        actions = []
        for i in range(len(state)):
            actions.append(np.random.choice(self.selection_matrix[:, i, state[i]].flatten(),
                                            p=self.policy_distribution[:, i, state[i]].flatten()))

        return actions
