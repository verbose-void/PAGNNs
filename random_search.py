"""
This module handles searching for the best network by randomly initializing networks and letting them rollout until they die.
"""

import numpy as np 
from graph_nn import GraphNN
from snake import Board as SnakeEnvironment, VALID_DIRECTIONS
from scipy.special import softmax


APPLE_REWARD = 1.3 # reverse weight_retention application

class SnakePopulation:

    def __init__(self, size, in_neurons, neurons, out_neurons, world_size=(10, 10)):
        # TODO: allow random amount of neurons
        self._size = size
        self._world_size = world_size
        self._in_neurons = in_neurons
        self._neurons = neurons
        self._out_neurons = out_neurons

        self.reset()


    def reset(self):
        self.networks = []
        self.environments = []
        for _ in range(self._size):
            nn = GraphNN(self._in_neurons, self._neurons, self._out_neurons)
            env = SnakeEnvironment(self._world_size, \
                    lambda env: self.get_direction_prediction(nn), \
                    lambda _: nn.reward(APPLE_REWARD))

            self.networks.append(nn)
            self.environments.append(env)


    def get_direction_prediction(self, nn):
        # output = softmax(nn.extract_output())
        output = nn.extract_output()
        i = np.argmax(output)
        direction = VALID_DIRECTIONS[i]
        print(direction)
        return direction


    def step_all(self):
        for nn, env in zip(self.networks, self.environments):
            # print(nn)
            # print(env.reward())
            env.rollout(1, draw=False)
            break
        



if __name__ == '__main__':
    population_size = 5
    out_neurons = len(VALID_DIRECTIONS) # number of possible actions 

    population = SnakePopulation(population_size, 1, 5, out_neurons)
    population.step_all()

