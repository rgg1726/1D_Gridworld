import numpy as np
from scipy.special import logsumexp


class Solver:
    """
    Compute the algorithm
    """
    def __init__(self, eta, gamma, alpha):
        """
        :param eta:
        :param gamma:
        :param alpha:
        """
        self.eta = eta
        self.gamma = gamma
        self.alpha = alpha

    def value(self, s, pi, q_table):

        value = 1 / self.alpha * logsumexp(self.alpha * q_table[:, s] + np.log(pi[:, s]), axis=0)

        return value

    def delta_q(self, r, s_prime, q, pi, q_table):
        delta_q = r + self.gamma * self.value(s_prime, pi, q_table) - q

        return delta_q

    def loss(self, q, results, q_table, pi, v0, size):
        L = len(results)
        s_prime = [item[2] for item in results]
        r = [item[3] for item in results]

        loss = 1 / self.eta * logsumexp(self.eta * self.delta_q(r, s_prime, q, pi, q_table) + np.log(1 / L)) + \
            (1 - self.gamma) * np.dot(v0, self.value(np.arange(size), pi, q_table))

        return loss
