from environment import State
from agent import Agent
from objective import Solver
import numpy as np
from scipy.optimize import minimize

size = 5
n_agents = 2
actions = ("left", "right")
init_state_distribution = np.array([0, 0, 1, 0, 0])
rewards = {"left": (10, 1), "right": (1, 3)}
smoothing = False

eta = 10
gamma = 0.9
alpha = 10

state = State(size, n_agents, init_state_distribution, rewards)
agents = [Agent(i, size, n_agents, actions) for i in range(n_agents)]
solver = Solver(eta, gamma, alpha)

K = 10
L = 100
for k in range(K):
    for i in range(n_agents):
        state.reset()

        j = np.random.choice(n_agents)

        results = []
        l = 0
        while l < L:
            current_state = state.get_state()
            sample_actions = agents[j].choose_action(current_state)
            next_state = state.next_state(sample_actions)
            reward = state.give_reward()

            if sample_actions[i] == "l":
                sample_actions[i] = 0
            elif sample_actions[i] == "r":
                sample_actions[i] = 1

            results.append((current_state[i], sample_actions[i], next_state[i], reward[i]))

            state.is_end()

            l += 1

        pi = agents[j].policy_distribution[:, i, :]
        q_table = agents[i].q_values.flatten()

        bounds = [(-120, 120) for i in range(L)]

        minimizer = minimize(solver.loss, q_table, (results, pi, init_state_distribution, size, len(actions)))
        q_i = minimizer.x


        agents[i].policy_update(q_i, results, alpha)
        print(agents[i].policy_distribution[:, 1, :])
