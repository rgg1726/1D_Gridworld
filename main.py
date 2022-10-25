from environment import State
from agent import Agent

size = 5            # size of the grid
n_agents = 2        # number of agents
init_position = 2   # initial position of agents in the grid

state = State(size, n_agents, init_position)
agents = [Agent(i) for i in range(n_agents)]    # create a list of "agent" objects

num_iters = 10
state.show_grid()
for iter in range(num_iters):
    actions = []
    for agent in agents:
        actions.append(agent.choose_action())
    print(actions)
    state.next_state(actions)
    state.show_grid()