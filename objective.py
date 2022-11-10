import numpy as np


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

    def value(self, s_prime, pi, q_table):
        value = -1 / self.alpha * np.log(np.sum(pi[:, s_prime] * np.exp(-self.alpha * q_table[:, s_prime])))

        return value

    def delta_q(self, c, s_prime, q, pi, q_table):
        delta_q = c + self.gamma * self.value(s_prime, pi, q_table) - q

        return delta_q

    def loss(self, q, results, q_table, pi):
        L = len(results)
        s_prime = [item[0] for item in results]
        c = [item[1] for item in results]

        inside_sum = 0
        for l in range(L):
            inside_sum += np.exp(-self.eta * self.delta_q(c[l], s_prime[l], q[l], pi, q_table))

        loss = -1 / self.eta * np.log(1 / L * inside_sum) + (1 - self.gamma)

        return -loss
