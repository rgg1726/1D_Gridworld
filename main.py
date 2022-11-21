from environment import State
from agent import Agent
from objective import Loss
import numpy as np
import torch

n_states = 7
n_agents = 2
init_state_distribution = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.double)
rewards = {"left": (10, 1), "right": (7, 5)}

eta = 10
gamma = 0.9
alpha = 10

state = State(n_states, n_agents, init_state_distribution, rewards)
agents = [Agent(i, n_states, n_agents) for i in range(n_agents)]
loss = Loss(eta, gamma, alpha)

K = 20
L = 100

for k in range(K):
    for i in range(n_agents):
        state.reset()

        j = np.random.choice(n_agents)

        samples = np.zeros((4, n_agents, L), dtype=np.int64)
        for l in range(L):
            current_state = state.get_state()
            sample_actions = agents[j].choose_action(current_state)
            next_state = state.next_state(sample_actions)
            reward = state.give_reward()
            samples[:, :, l] = current_state[:, 0], sample_actions[:, 0], next_state[:, 0], reward
            state.is_end()

        pi = agents[j].pi.copy()
        q_table = agents[i].q_values.copy()

        pi_tensor = torch.from_numpy(pi)
        q_tensor = torch.from_numpy(q_table).requires_grad_()
        samples_tensor = torch.from_numpy(samples)
        v0_tensor = torch.from_numpy(init_state_distribution)

        optimizer = torch.optim.SGD([q_tensor], lr=alpha)
        optimizer.zero_grad()
        G = loss.forward(q_tensor, samples_tensor, pi_tensor, v0_tensor)
        G.backward()
        optimizer.step()

        agents[i].policy_update(q_table, alpha)

state.reset()
state.show_grid()
for i in range(4):
    current_state = state.get_state()
    sample_actions = agents[j].choose_action(current_state)
    next_state = state.next_state(sample_actions)
    state.show_grid()
    state.is_end()