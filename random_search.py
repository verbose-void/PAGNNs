"""
This module handles searching for the best network by randomly initializing networks and letting them rollout until they die.
"""

import numpy as np 
import tqdm
from copy import deepcopy
from graph_nn import GraphNN
from snake import Board as SnakeEnvironment, VALID_DIRECTIONS
from scipy.special import softmax


APPLE_REWARD = 1.3 # reverse weight_retention application

class SnakePopulation:

    def __init__(self, size, in_neurons, neurons, out_neurons, world_size=(10, 10), \
                 weight_retention=0.9, energy_retention=0.9):
        # TODO: allow random amount of neurons
        self._size = size
        self._world_size = world_size
        self._in_neurons = in_neurons
        self._neurons = neurons
        self._out_neurons = out_neurons

        # TODO
        self._weight_retention = weight_retention
        self._energy_retention = energy_retention
        self._check_death_frequency = 5 # check for network deaths every N steps

        self.reset()

    
    def reset_network(self, idx):
        nn = GraphNN(self._in_neurons, self._neurons, self._out_neurons)
        env = SnakeEnvironment(self._world_size, \
                lambda env: self.get_direction_prediction(nn), \
                lambda _: nn.reward(APPLE_REWARD))

        self.networks[idx] = nn
        self.environments[idx] = env

        if self._best_idx == idx:
            self._best_idx = 0
            self._best_score = float('-inf')


    def reset(self):
        self._best_idx = 0
        self._best_score = float('-inf')
        self._deaths = 0
        self._best_score_ever = float('-inf')
        self._best_network_copy = None
       
        self.networks = [None] * self._size
        self.environments = [None] * self._size

        for i in range(self._size):
            self.reset_network(i)


    def get_direction_prediction(self, nn):
        # output = softmax(nn.extract_output())
        output = nn.extract_output()
        i = np.argmax(output)
        direction = VALID_DIRECTIONS[i]
        return direction


    def run(self, steps, draw_best=True, use_tqdm=True):
        iterator = range(steps)
        if use_tqdm:
            iterator = tqdm.tqdm(iterator, desc='simulating population [size=%i]' % self._size, total=steps)

        if draw_best and use_tqdm:
            raise NotImplementedError('TODO')

        for step in iterator:
            population.step_all(draw_best=draw_best, check_dead=((step+1) % self._check_death_frequency == 0))


    def step_all(self, draw_best=True, check_dead=False):
        for i, (nn, env) in enumerate(zip(self.networks, self.environments)):
            # print(nn)
            # print(env.reward())

            X = np.asarray([env.get_observation()])
            nn.load_input(X)
            nn.step(self._weight_retention, self._energy_retention)

            if env.apples_eaten > self._best_score:
                self._best_score = env.apples_eaten
                self._best_idx = i

            if env.apples_eaten > self._best_score_ever:
                self._best_score_ever = env.apples_eaten
                self._best_network_copy = deepcopy(nn)

            draw = draw_best and self._best_idx == i
            env.rollout(1, draw=draw, suffix=\
                        'Network IDX: %i Deaths: %i' % (self._best_idx, self._deaths)) 

            if check_dead:
                if nn.is_dead():
                    # network is dead, reinitialize it and it's environment
                    self.reset_network(i)
                    self._deaths += 1 


    def metrics(self):
        return {
            'deaths': self._deaths,
            'best_score_ever': self._best_score_ever
        }


if __name__ == '__main__':
    population_size = 100 
    out_neurons = len(VALID_DIRECTIONS) # number of possible actions 

    weight_retention = 0.99
    energy_retention = 0.99

    population = SnakePopulation(population_size, 2, 10, out_neurons, weight_retention=weight_retention, \
                                 energy_retention=energy_retention)
    population.run(10000, draw_best=False) 
    print(population.metrics())
    best_nn = population._best_network_copy
    print(best_nn.latent_state)
    print(best_nn.graph_weights)




