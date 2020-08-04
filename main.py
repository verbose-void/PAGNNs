from PANN.pann import PANN
from experiments.graph_ffnn import _step

import numpy as np

if __name__ == '__main__':
    pann = PANN(10, 1, 1, initial_sparsity=0.0001)
    print(pann)
    X = np.random.uniform(low=0, high=10, size=(1, 1))

    y = pann(X, num_steps=3)
    print(y)

    # state = np.random.uniform(low=0, high=10, size=(10, 10))
    # print(_step(X, state))
    # print('\n')
    # print(X.T.dot(state).T)
