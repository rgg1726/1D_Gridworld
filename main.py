from environment import State
from agent import Agent
import random

size = 5            # size of the grid
n_agents = 2        # number of agents
actions = ["left", "right"]
init_position = 2   # initial position of agents in the grid

state = State(size, n_agents, init_position)
agents = [Agent(i, size, n_agents, actions) for i in range(n_agents)]    # create a list of "agent" objects

num_iters = 1
for iter in range(num_iters):
    for agent in agents:
        j = random.choice(agents)
        samples = []
        i = 0
        while i < 5:
            current_state = state.get_state()
            sample_actions = j.choose_action(current_state)
            next_state = state.next_state(sample_actions).tolist()
            samples.append((current_state, sample_actions, next_state))
            i += 1
        print(samples)