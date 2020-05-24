

class GraphNN:

    def __init__(self, in_neurons, neurons, out_neurons):
        """
        Similar to GraphFFNN, however this class isn't forced to follow a Feed-Forward structure.
        """

        if in_neurons + out_neurons > neurons:
            raise Exceptions('in/out neuron sum must be <= total number of neurons %i+%i>%i' % (in_neurons, out_neurons, neurons))

        self._in_neurons = in_neurons
        self._out_neurons = out_neurons
        self._neurons = neurons

        self.graph_weights = np.zeros((self.neurons, self.neurons))

        """
        Weight initialization methods:

        1. Scaling Sparsity:
            One way of initializing weights is to provide a sparsity value. The higher the sparsity vaulue, the lower the chance of allowing a synapse at some position. If the sparsity value is 1, the weights will all be 0s. If the sparsity value is 0, then none of the weights will be 0, therefore 0 sparsity. 0.5 means that there is a 50% chance that a given synaptic value (or weight) is != 0.

        2. Scaling Sparsity + Forced Connections:
            This method is basically the same as the scaling sparsity method, however it can force input/output neurons to have connections. For example, a feed forward neural network 
        """

if __name__ == '__main__':
    nn = GraphNN(1, 5, 1)
    print(nn)
