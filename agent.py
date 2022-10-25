import numpy as np
import random


class Agent(object):
    def __init__(self, id):
        self.id = id

    # def choose_action(self, state):
    #     """
    #     :param state: global state value
    #     choose the action for the agent based on the global state
    #     :return:action that the agent should take
    #     """

    def choose_action(self):
        return random.choice(["left", "right"])