from environment import State
from agent import Agent
from objective import Solver
import random
import numpy as np
from scipy.optimize import minimize

size = 5
n_agents = 2
actions = ("left", "right")
init_position = 2
rewards = {"left": (10, 1), "right": (1, 3)}

eta = 0.5
gamma = 1
alpha = 0.5

state = State(size, n_agents, init_position, rewards)
agents = [Agent(i, size, n_agents, actions) for i in range(n_agents)]
solver = Solver(eta, gamma, alpha)

K = 1
L = 100
H = 5
for k in range(K):
    for i in range(n_agents):
        state.reset()

        j = random.randrange(n_agents)

        results = []
        q = []
        l = 0
        while l < L:
            current_state = state.get_state()
            sample_actions = agents[j].choose_action(current_state)
            next_state = state.next_state(sample_actions)
            reward = state.give_reward()

            results.append((next_state[i], reward[i]))

            if sample_actions[i] == "l":
                sample_actions[i] = 0
            elif sample_actions[i] == "r":
                sample_actions[i] = 1
            q.append(agents[i].q_values[sample_actions[i], current_state[i]])

            state.is_end()

            l += 1

        pi = agents[j].policy_distribution[:, i, :]
        q_table = agents[i].q_values
        bounds = [(-10, 10) for i in range(L)]

        q_i = minimize(solver.loss, np.asarray(q), (results, q_table, pi), bounds=bounds)
        print(q_i)


