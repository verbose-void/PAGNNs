import numpy as np
from graph_ffnn import _step


def _init_weights(W, s, low=-3, high=3):
    if np.random.uniform() > s:
        return np.random.uniform(low=low, high=high+1)
    return 0.0 
_init_weights = np.vectorize(_init_weights)


class GraphNN:

    def __init__(self, in_neurons, neurons, out_neurons, sparsity_value=0.5, force_input_connections=False, \
                 force_output_connections=False, dtype=np.float32):
        """
        Similar to GraphFFNN, however this class isn't forced to follow a Feed-Forward structure.
        """

        if in_neurons + out_neurons > neurons:
            raise Exceptions('in/out neuron sum must be <= total number of neurons %i+%i>%i' % (in_neurons, out_neurons, neurons))

        self._in_neurons = in_neurons
        self._out_neurons = out_neurons
        self._neurons = neurons
        self._dtype = dtype
        
        self.reset_latent_state()
        self.graph_weights = np.zeros((self._neurons, self._neurons), dtype=dtype)

        """
        Weight initialization methods:

        1. Scaling Sparsity:
            One way of initializing weights is to provide a sparsity value. The higher the sparsity vaulue, the lower the chance of allowing a synapse at some position. If the sparsity value is 1, the weights will all be 0s. If the sparsity value is 0, then none of the weights will be 0, therefore 0 sparsity. 0.5 means that there is a 50% chance that a given synaptic value (or weight) is != 0.

        2. Scaling Sparsity + Forced Connections (Inward/Outward):
            This method is basically the same as the scaling sparsity method, however it can force input and/or output neurons to have connections. For example, a feed forward neural network forces connections for input neurons to be fully connected to every neuron in the following layer. In FFNNs, the output neurons don't have forced outward connections, however they do have forced INWARD connections, meaning each output neuron is inwardly connected to every neuron in the previous layer.
        """

        if force_input_connections or force_output_connections:
            raise NotImplementedError('TODO')

        self.graph_weights = _init_weights(self.graph_weights, sparsity_value)

    def reset_latent_state(self):
        self.latent_state = np.zeros((self._neurons, self._neurons), dtype=self._dtype)

    def is_dead(self, threshold=1e-1):
        """
        A dead network is defined by a network who's weight's absolute value sum is < some threhold value (ie. 1e-5).
        This means that the network will not provide any outputs no matter how many steps you take.
        """

        return np.sum(np.abs(self.graph_weights)) <= threshold

    def load_input(self, X, steps=1):
        """
        Load the input data "X" into neuron format and store in the network's "latent state". This is NOT a forward pass, it is also not a state step, it simply loads the data into the latent space. 

        In order to propagate, call GraphNN.step(...).
        """
        
        # convert input data into neuron format
        x = X[0] # TODO: handle batch
        D = len(x)
        self.latent_state = np.pad(x[0], (0, self._neurons-D))
        self.latent_state = (self.graph_weights.T * self.latent_state).T


    def step(self, weight_retention=0.9, energy_retention=0.9):
        self.latent_state = _step(self.graph_weights, self.latent_state) * energy_retention
        self.graph_weights *= weight_retention 


    def extract_output(self):
        Y = np.zeros(self._out_neurons)
        c = 0
        for i in range(self._out_neurons):
            Y[c] = self.latent_state[-i, -i]
            c += 1
        return Y


if __name__ == '__main__':
    input_features = 1 
    output_features = 2 
    neurons = 5
    nn = GraphNN(input_features, neurons, output_features, sparsity_value=0.5)
        
    # -print(nn)
    # print(nn.graph_weights, nn.graph_weights.dtype)

    X = np.random.randint(1000, size=(1, input_features))
    print('X:', X)



    print('Initial State')
    print(nn.latent_state.astype(np.int32))

    print('After input Loading')
    nn.load_input(X)
    print(nn.latent_state.astype(np.int32))
    
    max_steps = 1000
    for step in range(max_steps):
        # print('Step %i:' % step)
        nn.step(weight_retention=0.2, energy_retention=0.2)

        # print(nn.latent_state.astype(np.int32))
        print('step %i output: %s' % (step, str(nn.extract_output())))

        if nn.is_dead():
            print('Lasted %i steps.' % step)
            break


