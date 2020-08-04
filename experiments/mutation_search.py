import numpy as np
from copy import deepcopy
from snake import VALID_DIRECTIONS
from random_search import RandomSnakePopulation


def mutate_weights(W, max_magnitude=5):
    # bump weights. if a weight is 0, that network will gain a synapse
    i, j = np.random.randint(low=0, high=W.shape[0], size=2) # square matrix
    bump = np.random.uniform(low=-max_magnitude, high=max_magnitude+1) 
    W[i, j] += bump


def mutate_hyperparams(hyperparams):
    # TODO
    pass


class MutationSnakePopulation(RandomSnakePopulation):

    def __init__(self, hyperparams_mutate_chance=0.5, weights_mutate_chance=0.5, **kwargs):
        super().__init__(**kwargs)

        self.hyperparams_mutate_chance = hyperparams_mutate_chance
        self.weights_mutate_chance = weights_mutate_chance


    def get_new_network(self, idx, hyperparams, is_initial):
        nn = super().get_new_network(idx, hyperparams, is_initial)

        if is_initial:
            return nn
            
        # when a network dies, choose a random network (weighted by their performance) and do some mutation to it
        # TODO: right now, only taking the best network
        # random_idx = np.random.randint(self._size)
        # while random_idx == idx:
            # random_idx = np.random.randint(self._size) # force pick a different one (not efficient)
        random_idx = self._best_idx # TODO pick actually random

        # grab the reference network
        new_nn = deepcopy(self.networks[random_idx])
        new_hp = self.hyperparams[random_idx]
        new_nn.revert_to_initial()

        # do mutations
        if np.random.uniform() < self.hyperparams_mutate_chance:
            mutate_hyperparams(new_hp)
        if np.random.uniform() < self.weights_mutate_chance:
            # do mutation n times (random choice)
            n = np.random.randint(low=1, high=10)
            for _ in range(n):
                mutate_weights(new_nn.graph_weights)

        return nn # no need to return hyperparams; they are modified by reference 



if __name__ == '__main__':
    world_size = (10, 10)
    size = 100
    out_neurons = len(VALID_DIRECTIONS)
    in_neurons = 2

    # hyperparams
    neurons = (in_neurons + out_neurons, 15)
    weight_retention = (0.1, 1)
    energy_retention = (0.1, 1)
    sparsity_value = (0, 0.9)

    population = MutationSnakePopulation(size=size, in_neurons=in_neurons, neurons=neurons, out_neurons=out_neurons, \
                                         weight_retention=weight_retention, energy_retention=energy_retention, \
                                         sparsity_value=sparsity_value, world_size=world_size)
    # population.step_all(draw_best=False)
    population.run(100, draw_best=False)

    print(population.metrics())
