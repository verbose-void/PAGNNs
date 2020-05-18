
import numpy as np


class GraphFFNN:

    def __init__(self, input_dim, hidden_units, output_dim, use_bias=True):
        self.input_dim = input_dim

        # Define the weights and biases in the normal neural network domain.
        self.W = []
        self.B = []
        prev_units = input_dim
        self._num_neurons = input_dim
        for units in list(hidden_units) + [output_dim]:
            self._num_neurons += units
            max_magnitude = np.sqrt(prev_units * units)
            
            self.W.append(np.random.uniform(low=-max_magnitude, high=max_magnitude, size=(prev_units, units)))
            
            if use_bias:
                self.B.append(np.random.uniform(low=-max_magnitude, high=max_magnitude, size=(units, )))
            else:
                self.B.append(np.zeros(shape=(units,)))

            prev_units = units

        # Define the weights in the graph-based neural network domain.
        
        # Input data is also considered to be made up of neurons. To understand this better, I would like to bring in an analogy to the human nervous system. Our 
        # nervous system is made up of trillions of neurons. What exactly a neuron is is a transmitter of electrical signals. You can think of the nervous system
        # as an extension of our brains, tendrils running out to gather data about our immediate environment. Therefore, our brain takes in inputs through  neurons
        # whether that be directly or indirectly through our environmental sensors.

        # Traditional thinking in the ANN research community is to conceptually separate these environmental sensors. However, if Monism has taught us anything,
        # it's that we are one with our environment, one with ourself, and one with the universe. So why should our networks be treated any differently?

        # Weights in this representation should be thought of as the synapses in our brains. Though this class is specific to feed forward neural networks, the
        # same structure should generally apply to any architecture of neurons and synapses.

        # Another point I'd like to make, is that in this representation the network has a state. This state is a representation of the latent variables at a given
        # time step. In a feed forward neural network, the forward pass' number of states are relative to the number of layers there are in the network. For example,
        # a network with 6 layers will have 7 states in the forward pass. 1=input loading, 2=layer1 pass, 3=layer2 pass, ..., 7=layer5 pass. After it propagates
        # through each state (assuming there are no occilations in the architecture, like a feed forward neural network), the "output" will be stored in the graph
        # nodes representing the output dimensions. This is kept as a batch size of 1 to understand conceptually.

        self.graph_weights = np.zeros((self._num_neurons, self._num_neurons)) # Matrix graph representation for our weights
        
        # initialize graph_weights with the same weights as the normal mode
        neuron_idx = 0
        for W, B in zip(self.W, self.B):
            N, D = W.shape
            self.graph_weights[neuron_idx:neuron_idx+N, neuron_idx+N:neuron_idx+D+N] = W
            neuron_idx += N

        # set last neurons to be connected to themselves (for now) TODO: change this, doesn't make sense for general application
        for i in range(output_dim):
            self.graph_weights[-(i+1), -(i+1)] = 1

    def forward(self, X, mode='normal'):
        if mode not in ('normal', 'graph'):
            raise Exception('Forward mode must be normal or graph.')
        
        if mode == 'normal':
            # TODO: Currently only linear transforms. We need to introduce non-linearity
            
            Z = X.T
            for W, B in zip(self.W, self.B):
                Z = np.dot(W.T, Z)
                Z += np.transpose([B])
            Y = Z.T

        elif mode == 'graph':
            # TODO: handle batch later
            state = X[0]
            D = len(state)
            state = np.pad(state, (0, self._num_neurons-D))
            state = self.graph_weights * np.transpose([state])

            state = self.graph_step(state)
            state = self.graph_step(state)
            state = self.graph_step(state)
            state = self.graph_step(state)
            Y = state

        return Y

    def graph_step(self, latent_graph_state):
        # TODO: make more efficient, for now we're traversing the graph nodes to their next edges
        
        next_state = np.zeros(latent_graph_state.shape)

        for i, neuron in enumerate(latent_graph_state):
            next_state += self.graph_weights * np.transpose([neuron])
            
        return next_state 

    def __str__(self):
        return str(self._num_neurons)



if __name__ == '__main__':
    x_features = 2
    gnn = GraphFFNN(x_features, (3, 5, 3), 1, use_bias=False)
    print('Number of neurons:', gnn._num_neurons)
    
    # create input data
    X = np.random.randint(1000, size=(1, x_features))
    print('X:', X)
    print()
    
    # test normal neural network domain inference
    y = gnn.forward(X, mode='normal')
    print('Normal output:')
    print(y)

    y_graph = gnn.forward(X, mode='graph')
    print('Graph output:')
    print(y_graph)

    # print()
    # print(gnn.graph_weights)
    # print()
