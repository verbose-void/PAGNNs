
import numpy as np


class GraphFFNN:

    def __init__(self, input_dim, hidden_units, output_dim):
        self.input_dim = input_dim
        self.W = []
        self.B = []
        
        prev_units = input_dim
        self._num_neurons = 0
        for units in list(hidden_units) + [output_dim]:
            self._num_neurons += units
            max_magnitude = np.sqrt(prev_units * units)
            
            self.W.append(np.random.uniform(low=-max_magnitude, high=max_magnitude, size=(prev_units, units)))
            self.B.append(np.random.uniform(low=-max_magnitude, high=max_magnitude, size=(units, )))

            prev_units = units

    def forward(self, X, mode='normal'):
        if mode not in ('normal', 'graph'):
            raise Exception('Forward mode must be normal or graph.')
        
        if mode == 'normal':
            # TODO: Currently only linear transforms. We need to introduce non-linearity
            
            Z = X.T
            for i, (W, B) in enumerate(zip(self.W, self.B)):
                Z = np.dot(W.T, Z)
                Z += np.transpose([B])
            Y = Z.T

        elif mode == 'graph':
            pass

        return Y

    def __str__(self):
        return str(self._num_neurons)



if __name__ == '__main__':
    gnn = GraphFFNN(1, (3, 5, 3), 1)
    y = gnn.forward(np.random.randint(1000, size=(3, 1)))
    print(y)
