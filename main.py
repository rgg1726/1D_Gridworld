import numpy as np
from environment import State
from agent import Agent
import sys
"""
environment needs to reset
environment needs to "step" with action, if actions are the same
environment needs to sample trajectories
policy i needs action i and global state

for each iteration t = 1.....T
    for each agent n = 1.....N


"""

size = 3
n_agents = 2
init_position = 1
left_rewards = np.array([[10], [2]])
right_rewards = np.array([[1], [3]])

state = State(size, n_agents, init_position, left_rewards, right_rewards)
agents = Agent(n_agents, state)

agents.state_values[:, [0]] = left_rewards
agents.state_values[:, [size - 1]] = right_rewards

print(agents.get_state_values())

i = 1
while i < 3:
    agents.value_iteratation(state.get_state()[0])
    state.next_state(agents.choose_action(state.get_state()[0]))
    print(agents.get_state_values())
    print(agents.actions)
    i += 1