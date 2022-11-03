import numpy as np


class Solver:
    def __init__(self, eta, gamma, alpha, samples, q, policy_distribution):
        self.eta = eta
        self.gamma = gamma
        self.alpha = alpha
        self.samples = samples
        self.q = q
        self.policy_distribution = policy_distribution

    def loss(self):
        L = len(self.samples)

        summation = 0
        for l in range(L):
            v = (-1 / self.alpha) * np.log(sum(self.policy_distribution[:, self.samples[l, 2]] * np.exp(-self.alpha * self.q[:, self.samples[l, 2]])))
            delta_q = self.samples[l, 3] + self.gamma * v - self.q[self.samples[l, 1], self.samples[l, 0]]
            summation += np.exp(-self.eta * delta_q)

        loss = -1 / self.eta * np.log(summation / L)

        return loss
