import torch
import torch.nn as nn

import numpy as np

from math import ceil

import networkx as nx


def _create_sparsity_freeze_function(weight):
    M = (weight == 0).int()

    def _freeze_sparsity(grad):
        return grad * M

    return _freeze_sparsity


class AdjacencyMatrix(nn.Module):
    def __init__(self, n, input_neurons=0, output_neurons=0, sparsity=0):
        super(AdjacencyMatrix, self).__init__()

        if sparsity < 0 or sparsity > 1:
            raise ValueError('sparsity must be on the closed interval [0, 1]. got %i.' % sparsity)

        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.n = n
        self.state = None
        self.sparsity = sparsity

        # initialize weights
        self.weight = nn.Parameter(torch.ones((n, n)))
        self.reset_parameters()

        self.zero_state()


    @torch.no_grad()
    def reset_parameters(self, only_modify_hidden_weight=False):
        if self.output_neurons > 0:
            self.hidden_weight = self.weight[self.input_neurons:-self.output_neurons]
        else:
            self.hidden_weight = self.weight[self.input_neurons:]

        weights_to_initialize = self.weight
        if only_modify_hidden_weight:
            weights_to_initialize = self.hidden_weight
            
        # uniformly initialize weights
        nn.init.kaiming_uniform_(weights_to_initialize, mode='fan_in', nonlinearity='relu')

        # inject random sparsity
        _flat = weights_to_initialize.view(-1)
        indices_pool = torch.randperm(len(_flat))
        num_non_sparse_elems = min(len(indices_pool)-1, int(ceil(len(indices_pool)*self.sparsity)))
        random_indices = indices_pool[:num_non_sparse_elems]
        _flat[random_indices] = 0


    @torch.no_grad()
    def zero_state(self):
        if self.state is None:
            self.state = torch.zeros((self.n, self.n))


    @torch.no_grad()
    def load_input_neurons(self, x):
        D = len(x[0]) # TODO batch

        if D != self.input_neurons:
            raise ValueError('dimensionality of input data must be the same as number of input neurons. D=%i expected %i.' % \
                             (D, self.input_neurons))

        self.state = np.pad(x[0], (0, self.n-D))
        self.state = (self.weight.T * self.state).T


    def extract_output_neurons(self):
        Y = torch.zeros(self.output_neurons)
        c = 0
        for i in range(self.output_neurons, 0, -1):
            Y[c] = self.state[-i, -i]
            c += 1
        return Y 


    def step(self, n=1):
        for _ in range(n):
            next_state = torch.zeros(self.state.shape)

            for i in range(self.state.shape[0]):
                # next_state += self.graph_weights * np.transpose([neuron]) # this method is MUCH slower
                next_state = next_state + (self.weight.T * self.state[i])

            self.state = next_state.T


    def forward(self, x, num_steps=1):
        self.load_input_neurons(x)
        self.step(n=num_steps)
        return self.extract_output_neurons()


    def extra_repr(self):
        return 'num_neurons=%i, input_neurons=%i, output_neurons=%i' % (self.n, self.input_neurons, self.output_neurons)


    def get_networkx_graph(self, return_color_map=True):
        W = self.weight.detach().numpy()
        G = nx.Graph(W)

        if not return_color_map:
            return G

        color_map = []
        for i, neuron in enumerate(W):
            # "input" neurons
            if i < self.input_neurons:
                color_map.append('green')

            # "hidden" neurons
            elif i >= (self.n - self.output_neurons):
                color_map.append('blue')

            # "output" neurons
            else:
                color_map.append('gray')

        return G, color_map


class PAGNN(nn.Module):
    def __init__(self, num_neurons, input_neurons, output_neurons, initial_sparsity=0.9, freeze_sparsity_gradients=True):
        super(PAGNN, self).__init__()

        if input_neurons + output_neurons > num_neurons:
            raise ValueError('number of allocated input & output neurons cannot add up to be greater than the number of total neurons. \
                              got %i (input) + %i (output) > %i (total).' % (input_neurons, output_neurons, num_neurons))

        self.n = num_neurons
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.structure_adj_matrix = AdjacencyMatrix(num_neurons, input_neurons=input_neurons, output_neurons=output_neurons, sparsity=initial_sparsity)

        if freeze_sparsity_gradients:
            sam_weight = self.structure_adj_matrix.weight
            self.structure_adj_matrix.weight.register_hook(_create_sparsity_freeze_function(sam_weight))
    

    def forward(self, x, num_steps=1):
        y = self.structure_adj_matrix(x, num_steps=num_steps)
        return y


    def extra_repr(self):
        return 'num_neurons=%i, input_neurons=%i, output_neurons=%i' % (self.n, self.input_neurons, self.output_neurons)


    def get_networkx_graph(self, return_color_map=True):
        return self.structure_adj_matrix.get_networkx_graph(return_color_map=return_color_map)
