"""
This module handles searching for the best network by randomly initializing networks and letting them rollout until they die.
"""

import numpy as np 
import tqdm
import pickle
from time import sleep
from copy import deepcopy
from graph_nn import GraphNN
from snake import Board as SnakeEnvironment, VALID_DIRECTIONS
from scipy.special import softmax


APPLE_REWARD = 1.3 # reverse weight_retention application

def _get_hyperparam(x, integer=False):
    if isinstance(x, tuple):
        if integer:
            return np.random.randint(x[0], x[1])
        return np.random.uniform(x[0], x[1])
    return x

class RandomSnakePopulation:

    def __init__(self, size, in_neurons, neurons, out_neurons, world_size=(10, 10), \
                 weight_retention=0.9, energy_retention=0.9, sparsity_value=0.5):
        """
        If neurons, weight_retention, or energy_retention are tuples of size 2, treat them as ranges.
        """

        self._size = size
        self._world_size = world_size
        self._in_neurons = in_neurons
        self._neurons = neurons
        self._out_neurons = out_neurons

        self._weight_retention = weight_retention
        self._energy_retention = energy_retention
        self._sparsity_value = sparsity_value

        self._check_death_frequency = 5 # check for network deaths every N steps

        self.reset()

    
    def reset_network(self, idx, initial_seed=None):
        # initialize hyper parameters. don't count as part of the initial random state
        hyperparams = {
            'weight_retention': _get_hyperparam(self._weight_retention),
            'energy_retention': _get_hyperparam(self._energy_retention),
            'neurons': _get_hyperparam(self._neurons, integer=True),
            'sparsity_value': _get_hyperparam(self._sparsity_value),
        }
        nn = GraphNN(self._in_neurons, hyperparams['neurons'], self._out_neurons, sparsity_value=hyperparams['sparsity_value'])         

        # set a random seed and log the state
        np.random.seed(initial_seed)
        random_state = np.random.get_state()

        env = SnakeEnvironment(self._world_size, \
                lambda env: self.get_direction_prediction(nn), \
                lambda _: nn.reward(APPLE_REWARD))

        self.networks[idx] = nn
        self.environments[idx] = env
        self.initial_random_states[idx] = deepcopy(random_state)
        self.current_random_states[idx] = deepcopy(random_state)
        self.hyperparams[idx] = hyperparams

        if self._best_idx == idx:
            self._best_idx = 0
            self._best_score = float('-inf')
            self._best_hyperparameters = None


    def reset(self):
        self._best_idx = 0
        self._best_score = float('-inf')
        self._deaths = 0
        self._best_score_ever = float('-inf')
        self._best_network_copy = None
        self._best_hyperparameters = None
        self._best_network_initial_random_state = None
        self._scores = []
       
        self.networks = [None] * self._size
        self.environments = [None] * self._size
        self.hyperparams = [None] * self._size
        self.initial_random_states = [None] * self._size
        self.current_random_states = [None] * self._size

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

        # after doing a run, store all current scores
        for env in self.environments:
            self._scores.append(env.apples_eaten)


    def step_all(self, draw_best=True, check_dead=False):
        for i, (nn, env, rstate, hyperparams) in enumerate(zip(self.networks, self.environments, self.current_random_states, \
                                                               self.hyperparams)):
            # BEFORE ANYTHING, SET RANDOM STATE
            np.random.set_state(rstate)
            
            X = np.asarray([env.get_observation()])
            nn.load_input(X)
            nn.step(hyperparams['weight_retention'], hyperparams['energy_retention'])

            if env.apples_eaten > self._best_score:
                self._best_score = env.apples_eaten
                self._best_idx = i

            if env.apples_eaten > self._best_score_ever:
                self._best_score_ever = env.apples_eaten
                self._best_network_copy = deepcopy(nn)
                self._best_network_initial_random_state = deepcopy(self.initial_random_states[i])
                self._best_hyperparameters = deepcopy(self.hyperparams[i])

            draw = draw_best and self._best_idx == i
            env.rollout(1, draw=draw, suffix=\
                        'Network IDX: %i Deaths: %i' % (self._best_idx, self._deaths)) 

            if check_dead:
                if nn.is_dead():
                    # network is dead, reinitialize it and it's environment
                    self.reset_network(i)
                    self._deaths += 1 
                    # log how many apples they ate before dying
                    self._scores.append(env.apples_eaten)

            # UPDATE RANDOM STATE
            self.current_random_states[i] = np.random.get_state()


    def metrics(self):
        return {
            'deaths': self._deaths,
            'best_score_ever': self._best_score_ever,
            'scores': self._scores
        }


if __name__ == '__main__':
    REPLAY = False
    output_file = 'best.pkl'
    
    world_size = (10, 10)
    population_size = 1000
    out_neurons = len(VALID_DIRECTIONS) # number of possible actions 
    in_neurons = 2

    # hyperparameter ranges
    weight_retention = (0.1, 1)
    energy_retention = (0.1, 1) 
    sparsity_value = (0, 0.9)
    neurons = (in_neurons + out_neurons, 25)


    if not REPLAY:
        population = RandomSnakePopulation(population_size, in_neurons, neurons, out_neurons, weight_retention=weight_retention, \
                                     energy_retention=energy_retention, sparsity_value=sparsity_value, world_size=world_size)
        population.run(1000, draw_best=False) 
        # print(population.metrics())

        # get best found network & random state
        best_nn = deepcopy(population._best_network_copy)
        best_nn.revert_to_initial()
        best_nn_initial_random_state = deepcopy(population._best_network_initial_random_state)

        # save best network
        with open(output_file, 'wb') as f:
            pickle.dump((population._best_network_copy, population._best_network_initial_random_state, \
                         population._best_hyperparameters, population.metrics()), f)
            print('Saved best net to %s.' % output_file)
        
        print(population.metrics())
        hyperparams = population._best_hyperparameters
    else:
        # open network saved in the output_file
        with open(output_file, 'rb') as f:
            best_nn, best_nn_initial_random_state, hyperparams, metrics = pickle.load(f)
            best_nn.revert_to_initial() 
            
        population = RandomSnakePopulation(population_size, in_neurons, hyperparams['neurons'], out_neurons, \
                                     weight_retention=hyperparams['weight_retention'], \
                                     energy_retention=hyperparams['energy_retention'], \
                                     world_size=world_size)

    if input('Watch replay? [y]') != 'y':
        exit()

    # watch replay 
    np.random.set_state(best_nn_initial_random_state)
    env = SnakeEnvironment(world_size, \
            lambda env: population.get_direction_prediction(best_nn), \
            lambda _: best_nn.reward(APPLE_REWARD))

    weight_retention = hyperparams['weight_retention']
    energy_retention = hyperparams['energy_retention']

    replay_sleep_length = 0.1
    while not best_nn.is_dead():
        X = np.asarray([env.get_observation()])
        best_nn.load_input(X)

        best_nn.step(weight_retention, energy_retention)
        env.rollout(1, draw=True, sleep_length=replay_sleep_length, suffix='[REPLAY]')


