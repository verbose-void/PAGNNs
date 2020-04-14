# representing a FEED FORWARD NEURAL NETWORK as a graph is fairly simple. The dimensionality is the number of neurons x the number of neurons
# NxN. Their connections are represented by 1 or 0. we will ignore the weights for now.

import numpy as np

def make_ffnn_from_graph(net, layer_dims):
    # requires a square matrix, and the layer_dims to sum up to the square root of the product dimensionality of the net matrix.

    assert net.shape[0] == net.shape[1]
    assert len(net.shape) == 2
    assert np.sum(layer_dims) == net.shape[0]

    c = 0
    for i, layer_N in enumerate(layer_dims):
        # loop through each layer
        # fill in connections with neurons in next layer
        if i < len(layer_dims)-1:
            next_layer_neuron_first = c + layer_N
            next_layer_neuron_last = c + layer_N + layer_dims[i+1]
            net[c:c+layer_N, next_layer_neuron_first:next_layer_neuron_last] = 1

        c += layer_N


def graph_forward_pass(net, W, X):
    # so how do we calculate the forward pass?

    # first, get the input neuron indices. Input neuron indices in this case, are those with a column sum of 0.
    input_neurons = np.sum(net, axis=0) == 0
    N_input_neurons = np.sum(input_neurons)

    assert X.shape[0] == N_input_neurons, 'Number of inputs must match dimensionality of input data.'

    print(W, X)


if __name__ == '__main__':
    neuron_N = 9
    nn = np.zeros((9, 9))

    make_ffnn_from_graph(nn, (3, 3, 3))
    print('Network as graph:')
    print(nn)
    print()

    # if the sum of your columns is equal to 0, this means you have no connections to you as a neuron, and therefore will be treated
    # as an input neuron. At least in the case of feed forward neural networks.

    W = np.random.randint(low=1, high=10, size=nn.shape) # using integers for easier visualization
    W = nn * W # remove all of the unnecessary weights (make it sparse)
    print('Weights:')
    print(W)

    # generate some input data
    X = np.array([1, 2, 3])

    graph_forward_pass(nn, W, X)

