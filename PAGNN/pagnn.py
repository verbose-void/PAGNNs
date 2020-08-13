import torch
import torch.nn as nn

import numpy as np

from math import ceil

import networkx as nx
from operator import itemgetter


def _create_sparsity_freeze_function(weight):
    M = (weight == 0)

    def _freeze_sparsity(grad):
        grad[M] = 0
        return grad

    return _freeze_sparsity


def _is_valid_structure(W, num_output_neurons):
    valid = True
    abs_W = torch.abs(W)
    num_neurons = abs_W.shape[0]

    for neuron_idx in range(num_neurons):
        is_output_neuron = neuron_idx >= (num_neurons - num_output_neurons)

        neuron_abs_W = abs_W[neuron_idx]
        inward_neuron_abs_W = abs_W[:, neuron_idx]

        if torch.sum(neuron_abs_W != 0) <= 0:
            # print('neuron %i has no outward connections.' % neuron_idx)
            valid = False
            break

        required_non_sparse_elems = 1 if is_output_neuron else 0
        if torch.sum(inward_neuron_abs_W != 0) <= required_non_sparse_elems:
            # print('neuron %i has no inward connections.' % neuron_idx)
            valid = False
            break

        if is_output_neuron:
            # make sure this neuron has a non-sparse self-connection
            if neuron_abs_W[neuron_idx] <= 0: # self connection weight
                valid = False
                break

    return valid


class AdjacencyMatrix(nn.Module):
    def __init__(self, n, input_neurons=0, output_neurons=0, sparsity=0, create_using=nx.DiGraph, graph_generator=None):
        super(AdjacencyMatrix, self).__init__()

        if sparsity < 0 or sparsity > 1:
            raise ValueError('sparsity must be on the closed interval [0, 1]. got %i.' % sparsity)

        self.graph_generator = graph_generator
        self.create_using = create_using
        self.sparsity = sparsity

        if self.graph_generator is not None and self.sparsity != 0:
            raise Exception('sparsity must be 0 if using a graph_generator.')

        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.n = n
        self.state = None

        self.reset_parameters()


    @torch.no_grad()
    def reset_parameters(self, only_modify_hidden_weight=False):
        """
        rules for injecting sparsity:
        1. every neuron must have at least 1 OUTWARD connection (no OUTWARD neuron weight vector can be 100% sparse)
        2. every neuron must have at least 1 INWARD connection (no INWARD neuron weight vector can be 100% sparse)
        3. every OUTPUT neuron must have it's self-connection be non-sparse. in other words, every output neuron MUST be
           connected to itself.

        notes:
        - note on 1 & 2: essentially this means that each ROW/COLUMN magnitudal summation MUST be > 0
        - the reason for rules 1 & 2 is if you had a 100% sparse neuron, it's a source of informational death and can be 
          simply removed by reducing the number of allocated neurons.
        """

        if self.graph_generator is None:
            self.weight = nn.Parameter(torch.ones((self.n, self.n)))

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
        else:
            G = self.graph_generator(self.n, create_using=self.create_using)
            adj_matrix = nx.linalg.graphmatrix.adjacency_matrix(G).todense()

            # connect output neurons to themselves
            for i in range(self.output_neurons):
                idx = -(i+1)
                adj_matrix[idx, idx] = 1

            self.weight = nn.Parameter(torch.ones((self.n, self.n)))
            nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
            self.weight *= torch.tensor(adj_matrix)


    @torch.no_grad()
    def load_input_neurons(self, x, retain_state=False):
        N, D = x.shape

        if D != self.input_neurons:
            raise ValueError('dimensionality of input data must be the same as number of input neurons. D=%i expected %i.' % \
                             (D, self.input_neurons))

        device = self.weight.device
        if retain_state and self.state is not None:
            state_clone = self.state.clone()
            state_clone[:, 0:D] = x
            self.state = state_clone
        else:
            self.state = torch.cat((x, torch.zeros((N, self.n-D), device=device)), dim=1) # BATCH PADDING
        
        # initial_state = torch.zeros((N, self.n, self.n), device=device)
        # for i, s in enumerate(self.state):
            # initial_state[i] = (self.weight.T * s).T
        # self.state = initial_state


    def extract_output_neurons(self):
        # OLD BATCHED CODE (unvectorized)
        """
        device = self.weight.device
        N = self.state.shape[0] # BATCH SIZE
        Y = torch.zeros((N, self.output_neurons), device=device)
        for n in range(N):
            c = 0
            for i in range(self.output_neurons, 0, -1):
                Y[n, c] = self.state[n, -i, -i]
                c += 1
        return Y 
        """

        return self.state[:, -self.output_neurons:]


    def step(self, n=1, energy_scalar=1):
        # OLD BATCHED CODE (unvectorized)
        """
        device = self.weight.device
        for _ in range(n):
            next_state = torch.zeros(self.state.shape, device=device)
            TODO vectorize w.r.t batch dimension
            for n in range(self.state.shape[0]): # batch dimension
                next_sample = torch.zeros(self.state.shape[1:], device=device)
                for i in range(next_sample.shape[0]): # feature dimension
                    # next_state += self.graph_weights * np.transpose([neuron]) # this method is MUCH slower
                    next_sample = next_sample + (self.weight.T * self.state[n, i])
                # TODO REMOVE
                print(next_sample.T)
                print(next_sample.dtype)
                old_sample = next_sample.T
            self.state = next_state * energy_scalar
        """

        # EQUIVALENT UNBATCHED CODE
        """
        next_state = torch.zeros(self.state[0].shape)

        for i in range(self.state[0].shape[0]):
            # next_state += self.graph_weights * np.transpose([neuron]) # this method is MUCH slower
            next_state = next_state + (self.weight.T * self.state[0][i])

        # self.state = next_state.T
        UNBATCHED_STATE = next_state.T
        """

        for _ in range(n):
            self.state = torch.matmul(self.weight, self.state.T).T * energy_scalar


    def forward(self, x, num_steps=1, use_sequence=False, energy_scalar=1):
        self.state = None # clear state

        if use_sequence:
            for d in range(x.shape[1]): # feature dimension (feed one "feature" at a time -- could be a multi-dimensional "feature" if a sequence input)
                tx = x[:, d]
                self.load_input_neurons(tx, retain_state=True) # since there is a temporal aspect, we care about the previous state
                self.step(n=num_steps, energy_scalar=energy_scalar)
        else:
            # load all features into the input neurons simultaneously
            self.load_input_neurons(x, retain_state=False) # since we don't care about the previous state, clear it
            self.step(n=num_steps, energy_scalar=energy_scalar)

        y = self.extract_output_neurons()
        return y


    def extra_repr(self):
        return 'num_neurons=%i, input_neurons=%i, output_neurons=%i' % (self.n, self.input_neurons, self.output_neurons)


    def get_networkx_graph(self, return_color_map=True):
        W = self.weight.cpu().detach().numpy()
        G = nx.DiGraph(W)

        if not return_color_map:
            return G

        color_map = []
        for i, neuron in enumerate(W):
            # "input" neurons
            if i < self.input_neurons:
                color_map.append('green')

            # "output" neurons
            elif i >= (self.n - self.output_neurons):
                color_map.append('blue')

            # "hidden" neurons
            else:
                color_map.append('gray')

        return G, color_map


    def draw_networkx_graph(self, mode='default'):
        G, color_map = self.get_networkx_graph(return_color_map=True)

        if mode == 'default':
            nx.draw(G, with_labels=True, node_color=color_map)
        elif mode == 'ego':
            node_and_degree = G.degree()
            (largest_hub, degree) = sorted(node_and_degree, key=itemgetter(1))[-1]
            hub_ego = nx.ego_graph(G, largest_hub)
            pos = nx.spring_layout(hub_ego)
            nx.draw(hub_ego, pos, node_color='b', node_size=50, with_labels=False)
            nx.draw_networkx_nodes(hub_ego, pos, nodelist=[largest_hub], node_size=300, node_color='r')
        elif mode == 'karate_club':
            nx.draw_circular(G, with_labels=True, node_color=color_map)
        elif mode == 'roget':
            UG = G.to_undirected()
            nx.draw_circular(UG, node_color=color_map, node_size=1, line_color='grey', linewidths=0, width=0.1)
        elif mode == 'football':
            nx.draw(G, node_color=color_map, node_size=50, line_color='grey', linewidths=0, width=0.1)
        elif mode == 'scaled_weights':
            degrees = np.array([G.degree[n] for n in range(self.n)])
            degrees = degrees - np.min(degrees)
            degrees = degrees / np.max(degrees)
            degrees *= 200
            degrees += 10
            weightings = np.abs(self.weight.cpu().detach().numpy().flatten())
            weightings = weightings - np.min(weightings)
            weightings = weightings / np.max(weightings)
            weightings *= 1
            weightings += 0.1
            nx.draw(G, node_color=color_map, node_size=degrees, width=weightings)
        else:
            raise Exception('mode %s not found.' % mode)


class PAGNN(nn.Module):
    def __init__(self, num_neurons, input_neurons, output_neurons, initial_sparsity=0, freeze_sparsity_gradients=True, \
                 graph_generator=None, create_using=nx.DiGraph, w2vec_model=None):
        super(PAGNN, self).__init__()

        if input_neurons + output_neurons > num_neurons:
            raise ValueError('number of allocated input & output neurons cannot add up to be greater than the number of total neurons. \
                              got %i (input) + %i (output) > %i (total).' % (input_neurons, output_neurons, num_neurons))

        self.graph_generator = graph_generator
        self.create_using = create_using
        self.n = num_neurons
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.structure_adj_matrix = AdjacencyMatrix(num_neurons, input_neurons=input_neurons, output_neurons=output_neurons, sparsity=initial_sparsity, graph_generator=self.graph_generator, create_using=self.create_using)

        if freeze_sparsity_gradients:
            sam_weight = self.structure_adj_matrix.weight
            self.structure_adj_matrix.weight.register_hook(_create_sparsity_freeze_function(sam_weight))

        self.embedding = None
        if w2vec_model is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(w2vec_model.wv.vectors), \
                                                          padding_idx=w2vec_model.wv.vocab['pad'].index)
    

    def forward(self, x, num_steps=1, use_sequence=False, energy_scalar=1):
        if self.embedding is not None:
            use_sequence = True
            x = self.embedding(x)

        return self.structure_adj_matrix(x, num_steps=num_steps, use_sequence=use_sequence, energy_scalar=energy_scalar)


    def extra_repr(self):
        return 'num_neurons=%i, input_neurons=%i, output_neurons=%i, device=%s' % \
                (self.n, self.input_neurons, self.output_neurons, self.structure_adj_matrix.weight.device)


    def get_networkx_graph(self, return_color_map=True):
        return self.structure_adj_matrix.get_networkx_graph(return_color_map=return_color_map)

    def draw_networkx_graph(self, mode='default'):
        return self.structure_adj_matrix.draw_networkx_graph(mode=mode)
