import torch
import torch.nn as nn
from torch import logsumexp


class Loss(nn.Module):
    """
    Compute the empirical logistic policy evaluation objective
    """
    def __init__(self, eta, gamma, alpha):
        """
        :param eta:
        :param gamma:
        :param alpha:
        """
        super(Loss, self).__init__()
        self.eta = eta
        self.gamma = gamma
        self.alpha = alpha

    def value(self, pi, q_table, s):
        """
        logisitic value function for n_agents
        :param pi: global policy distribution (array n_actions x n_agents x n_states)
        :param q_table: global Q value table (array n_actions x n_agents x n_states)
        :param s:global states (array n_agents x n_samples)
        :return: sum of values over agents (array 1 x n_samples)
        """
        q = 0
        p = 1
        n_agents = s.shape[0]
        for agent in range(n_agents):
            q += q_table[:, agent, s[agent, :]] # sum q over agents
            p *= pi[:, agent, s[agent, :]]      # product p over agents

        value = 1 / self.alpha * logsumexp(self.alpha * q + torch.log(p), dim=0)

        return value

    def delta_q(self, s, a, s_prime, r, q_table, pi):
        """
        logistic bellman error for n_agents
        :param s: global state (array n_agents x n_samples)
        :param a: global actions (array n_agents x n_samples)
        :param s_prime: global next states (array n_agents x n_samples)
        :param r: global rewards (array n_agents x n_samples)
        :param q_table: global Q values (array n_actions, n_agents, n_states)
        :param pi: global policy (array n_actions, n_agents, n_states)
        :return: logistic bellman error summed over agents (array 1 x n_samples)
        """
        q = 0
        n_agents = s.shape[0]
        for agent in range(n_agents):
            q += q_table[a[agent, :], agent, s[agent, :]]   # sum q over agents

        delta_q = r.sum(dim=0) + self.gamma * self.value(pi, q_table, s_prime) - q

        return delta_q

    def forward(self, q_table, samples, pi, v0):
        """
        empirical logistic policy evaluation  objective
        :param q_table: global Q values (array n_actions, n_agents, n_states)
        :param samples: set of sampled states, actions, next states, rewards (array 4 x n_agents x n_samples)
        :param pi: global policy (array n_actions, n_agents, n_states)
        :param v0: initial global state (array n_agents, n_states)
        :return: empirical logistic policy evaluation  objective to be minimized wrt q
        """
        n_states = pi.shape[2]
        n_agents = pi.shape[1]
        L = samples.shape[2]
        s = samples[0, :, :]
        a = samples[1, :, :]
        a[a == -1] = 0
        s_prime = samples[2, :, :]
        r = samples[3, :, :]

        # construct a tensor holding all possible states for all agents
        state_tensor = torch.zeros((n_agents, n_states), dtype=torch.long)
        for i in range(n_agents):
            state_tensor[i, :] = torch.arange(n_states)

        loss = 1 / self.eta * logsumexp(self.eta * self.delta_q(s, a, s_prime, r, q_table, pi) +
                                        torch.log(torch.tensor([1 / L])), dim=0) +\
                                        (1 - self.gamma) * torch.dot(v0, self.value(pi, q_table, state_tensor))

        return loss
