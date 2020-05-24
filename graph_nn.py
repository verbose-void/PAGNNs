import numpy as np

class GraphNN:

    def __init__(self, in_neurons, neurons, out_neurons, sparsity_value=0.5, force_input_connections=False, \
                 force_output_connections=False):
        """
        Similar to GraphFFNN, however this class isn't forced to follow a Feed-Forward structure.
        """

        if in_neurons + out_neurons > neurons:
            raise Exceptions('in/out neuron sum must be <= total number of neurons %i+%i>%i' % (in_neurons, out_neurons, neurons))

        self._in_neurons = in_neurons
        self._out_neurons = out_neurons
        self._neurons = neurons

        self.graph_weights = np.zeros((self._neurons, self._neurons))

        """
        Weight initialization methods:

        1. Scaling Sparsity:
            One way of initializing weights is to provide a sparsity value. The higher the sparsity vaulue, the lower the chance of allowing a synapse at some position. If the sparsity value is 1, the weights will all be 0s. If the sparsity value is 0, then none of the weights will be 0, therefore 0 sparsity. 0.5 means that there is a 50% chance that a given synaptic value (or weight) is != 0.

        2. Scaling Sparsity + Forced Connections (Inward/Outward):
            This method is basically the same as the scaling sparsity method, however it can force input and/or output neurons to have connections. For example, a feed forward neural network forces connections for input neurons to be fully connected to every neuron in the following layer. In FFNNs, the output neurons don't have forced outward connections, however they do have forced INWARD connections, meaning each output neuron is inwardly connected to every neuron in the previous layer.
        """

        if force_input_connections or force_output_connections:
            raise NotImplementedError('TODO')

if __name__ == '__main__':
    nn = GraphNN(1, 5, 1, sparsity_value=0.5)
        
    print(nn)
