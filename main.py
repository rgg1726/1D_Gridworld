from environment import State
from agent import Agent
from objective import Loss
import numpy as np
import torch
from matplotlib import pyplot as plt

# define environment and constants
n_states = 11
n_agents = 2
rewards = {"left": (9, 2), "right": (1, 4)}    # number of rewards must match number of agents
start_point = 6
init_state_distribution = np.zeros(n_states)
init_state_distribution[start_point - 1] = 1

eta = 4
gamma = 0.9
alpha = 4
lr = 0.01

UNIFORM_SAMPLING_j = True  # sample j uniformly from all agents
KEEP_j = False              # keep same j for sample collection
w = np.identity(n_agents)   # weight matrix for no communication

VALIDATION_ON = True    # True to plot rewards over time
SHOW_GRID = True        # True to show the results of the final learned policy

# initialize the environment and agents
state = State(n_states, n_agents, init_state_distribution, rewards)
agents = [Agent(i, n_states, n_agents) for i in range(n_agents)]
loss = Loss(eta, gamma, alpha)

# K training iterations, L samples taken in an iteration
K = 100
L = 100

if VALIDATION_ON:   # if validating results, store rewards per iter
    agent_rewards = np.zeros((K, n_agents))
for k in range(K):
    for i in range(n_agents):
        state.reset()

        if KEEP_j:
            if UNIFORM_SAMPLING_j:
                j = np.random.choice(n_agents)
            else:
                j = np.random.choice(n_agents, p=w[i, :])

        samples = np.zeros((4, n_agents, L), dtype=np.int64)    # 4 = len([s, a, s', r])
        for l in range(L):

            if not KEEP_j:
                if UNIFORM_SAMPLING_j:
                    j = np.random.choice(n_agents)
                else:
                    j = np.random.choice(n_agents, p=w[i, :])

            # collect samples
            current_state = state.get_state()
            sample_actions = agents[j].choose_action(current_state)
            next_state = state.next_state(sample_actions)
            reward = state.give_reward()
            samples[:, :, l] = current_state[:, 0], sample_actions[:, 0], next_state[:, 0], reward

            state.is_end()  # reset state if end is reached

        if KEEP_j:
            pi = agents[j].pi.copy()
        else:
            pi = agents[i].pi.copy()
        q_table = agents[i].q_values.copy()

        pi_tensor = torch.from_numpy(pi)
        q_tensor = torch.from_numpy(q_table).requires_grad_()
        samples_tensor = torch.from_numpy(samples)
        v0_tensor = torch.from_numpy(init_state_distribution)

        # find argmin of loss wrt q
        optimizer = torch.optim.SGD([q_tensor], lr=lr)
        optimizer.zero_grad()
        for _ in range(10):
            G = loss.forward(q_tensor, samples_tensor, pi_tensor, v0_tensor)
            G.backward()
            optimizer.step()

        agents[i].policy_update(q_table, alpha)

    if VALIDATION_ON:
        state.reset()
        for i in range(L):
            current_state = state.get_state()
            agent = np.random.choice(n_agents)
            sample_actions = agents[agent].choose_action(current_state)
            next_state = state.next_state(sample_actions)
            agent_rewards[k, :] += state.give_reward()
            state.is_end()

if VALIDATION_ON:
    normalizing_factor = (start_point - 1) / L / max(sum(rewards["left"]), sum(rewards["right"]))
    plt.plot(agent_rewards.sum(axis=1) * normalizing_factor, label='global reward')
    for i in range(n_agents):
        plt.plot(agent_rewards[:, i] * normalizing_factor, label='agent {}'.format(i + 1))
    plt.xlabel("iteration")
    plt.ylabel("normalized reward over 100 steps")
    plt.legend(loc="upper right")
    plt.title("n_states = {}, n_agents = {}".format(n_states, n_agents))
    plt.show()

if SHOW_GRID:
    # try the learned policies
    state.reset()
    state.show_grid()
    for _ in range(10):
        current_state = state.get_state()
        sample_actions = agents[0].choose_action(current_state)
        next_state = state.next_state(sample_actions)
        state.show_grid()
        state.is_end()
