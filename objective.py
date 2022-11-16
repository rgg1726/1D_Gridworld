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

    def value(self, pi, q_table, size):

        s = np.arange(size)
        q_table = q_table.reshape(-1, size)

        value = 1 / self.alpha * logsumexp(self.alpha * q_table[:, s] + np.log(pi[:, s]), axis=0)


        return value

    def delta_q(self, r, q_table, pi, sampling_s, sampling_s_prime, size):
        delta_q = r + self.gamma * np.dot(sampling_s_prime, self.value(pi, q_table, size)) - np.dot(sampling_s, q_table)

        return delta_q

    def loss(self, q_table, results, pi, v0, size, n_actions):
        L = len(results)
        s = [item[0] for item in results]
        a = [item[1] for item in results]
        s_prime = [item[2] for item in results]
        r = [item[3] for item in results]

        sampling_matrix_s = np.zeros((L, size * n_actions))
        for j, item in enumerate(results):
            sampling_matrix_s[j, item[1] * size + item[0]] = 1

        sampling_matrix_s_prime = np.zeros((L, size))
        for j, item in enumerate(results):
            sampling_matrix_s_prime[j, item[2]] = 1



        loss = 1 / self.eta * logsumexp(self.eta * self.delta_q(r, q_table, pi, sampling_matrix_s, sampling_matrix_s_prime, size) + np.log(1 / L)) + \
            (1 - self.gamma) * np.dot(v0, self.value(pi, q_table, size))

        return loss
