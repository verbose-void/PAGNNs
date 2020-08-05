import torch
import torch.nn.functional as F

from PANN.pann import PANN
from experiments.graph_ffnn import _step

import numpy as np

if __name__ == '__main__':
    pann = PANN(10, 1, 1, initial_sparsity=0.5)
    print(pann)
    X = torch.tensor(np.random.uniform(low=0, high=10, size=(1, 1)))
    T = torch.tensor(np.random.uniform(low=0, high=10, size=(1, 1)), dtype=torch.float)

    print(pann.structure_adj_matrix.weight)

    optimizer = torch.optim.Adam(pann.parameters(), lr=0.01)
    for epoch in range(500):
        optimizer.zero_grad()
        with torch.enable_grad():
            y = pann(X, num_steps=3).unsqueeze(-1)
            loss = F.mse_loss(y, T)

        # print(loss.item(), T[0].item(), y[0].item())
        loss.backward()
        optimizer.step()
        # print(pann.structure_adj_matrix.weight.grad)

    print(loss.item(), T[0].item(), y[0].item())
    print(pann.structure_adj_matrix.weight)

    # state = np.random.uniform(low=0, high=10, size=(10, 10))
    # print(_step(X, state))
    # print('\n')
    # print(X.T.dot(state).T)
