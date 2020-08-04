import torch
import torch.nn as nn

from math import ceil


class AdjacencyMatrix(nn.Module):
    def __init__(self, n, input_neurons=0, output_neurons=0, sparsity=0):
        super(AdjacencyMatrix, self).__init__()

        if sparsity < 0 or sparsity > 1:
            raise ValueError('sparsity must be on the closed interval [0, 1]. got %i.' % sparsity)

        # initialize weights
        self.weight = torch.ones((n, n))
        if output_neurons > 0:
            self.hidden_weight = self.weight[input_neurons:-output_neurons]
        else:
            self.hidden_weight = self.weight[input_neurons:]

        # uniformly initialize weights
        nn.init.kaiming_uniform_(self.hidden_weight, mode='fan_in', nonlinearity='relu')

        # inject random sparsity
        _flat_hidden_weight = self.hidden_weight.view(-1)
        indices_pool = torch.randperm(len(_flat_hidden_weight))
        num_non_sparse_elems = min(len(indices_pool)-1, int(ceil(len(indices_pool)*sparsity)))
        random_indices = indices_pool[:num_non_sparse_elems]
        _flat_hidden_weight[random_indices] = 0

        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.n = n


    def forward(self, x):
        pass


    def extra_repr(self):
        return 'num_neurons=%i, input_neurons=%i, output_neurons=%i' % (self.n, self.input_neurons, self.output_neurons)


class PANN(nn.Module):
    def __init__(self, num_neurons, input_neurons, output_neurons, initial_sparsity=0.9):
        super(PANN, self).__init__()

        if input_neurons + output_neurons > num_neurons:
            raise ValueError('number of allocated input & output neurons cannot add up to be greater than the number of total neurons. \
                              got %i (input) + %i (output) > %i (total).' % (input_neurons, output_neurons, num_neurons))

        self.n = num_neurons
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.weight = AdjacencyMatrix(num_neurons, input_neurons=input_neurons, output_neurons=output_neurons, sparsity=initial_sparsity)
    

    def forward(self, x):
        pass


    def extra_repr(self):
        return 'num_neurons=%i, input_neurons=%i, output_neurons=%i' % (self.n, self.input_neurons, self.output_neurons)
