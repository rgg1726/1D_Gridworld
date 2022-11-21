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
        q = 0
        p = 1
        for agent in range(s.shape[0]):
            q += q_table[:, agent, s[agent, :]]
            p *= pi[:, agent, s[agent, :]]

        value = 1 / self.alpha * logsumexp(self.alpha * q + torch.log(p), dim=0)

        return value

    def delta_q(self, s, a, s_prime, r, q_table, pi):
        q = 0
        for agent in range(s.shape[0]):
            q += q_table[a[agent, :], agent, s[agent]]

        delta_q = r.sum(dim=0) + self.gamma * self.value(pi, q_table, s_prime) - q

        return delta_q

    def forward(self, q_table, samples, pi, v0):
        n_states = pi.shape[2]
        L = samples.shape[2]
        s = samples[0, :, :]
        a = samples[1, :, :]
        s_prime = samples[2, :, :]
        r = samples[3, :, :]
        for l in range(L):
            for agent in range(2):
                if a[agent, l] == -1:
                    a[agent, l] = 0

        state_tensor = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]])

        loss = 1 / self.eta * logsumexp(self.eta * self.delta_q(s, a, s_prime, r, q_table, pi) +
                                        torch.log(torch.tensor([1 / L])), dim=0) +\
                                        (1 - self.gamma) * torch.dot(v0, self.value(pi, q_table, state_tensor))

        return loss
