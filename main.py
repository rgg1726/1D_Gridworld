from environment import State
from agent import Agent
import random

size = 5           # size of the grid
n_agents = 2        # number of agents
actions = ("left", "right")
init_position = 2   # initial position of agents in the grid
num_samples = 5
rewards = {"left": (10, 1), "right": (1, 3)}

state = State(size, n_agents, init_position, rewards)
agents = [Agent(i, size, n_agents, actions) for i in range(n_agents)]    # create a list of "agent" objects

num_iters = 1
for iter in range(num_iters):
    for agent in range(len(agents)):
        state.reset()
        j = random.choice(agents)
        samples = []
        i = 0
        while i < num_samples:
            current_state = state.get_state()
            sample_actions = j.choose_action(current_state)
            next_state = state.next_state(sample_actions)
            reward = state.give_reward()[agent]
            samples.append((current_state[agent], sample_actions[agent], next_state[agent], reward))

            state.is_end()
            i += 1

        q = agents[agent].q_values
        policy_distribution = agents[agent].policy_distribution[:, agent, :]
        print(samples)
