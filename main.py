import torch
import torch.nn.functional as F

from PAGNN.pagnn import PAGNN

import numpy as np


def count_epochs_to_converge(pagnn, X, T, num_steps, thresh=1e-4, max_epochs=1000):
    optimizer = torch.optim.Adam(pagnn.parameters(), lr=0.01)
    num_epochs = 0
    loss_value = 1
    while loss_value > 0.0001:
        optimizer.zero_grad()
        with torch.enable_grad():
            y = pagnn(X, num_steps=num_steps).unsqueeze(-1)
            loss = F.mse_loss(y, T)

        loss.backward()
        optimizer.step()
        loss_value = loss.item()
        num_epochs += 1

        if num_epochs >= max_epochs:
            break

    return num_epochs

if __name__ == '__main__':
    pagnn = PAGNN(10, 1, 1, initial_sparsity=0.5)
    print(pagnn)
    X = torch.tensor(np.random.uniform(low=0, high=10, size=(1, 1)))
    T = torch.tensor(np.random.uniform(low=0, high=10, size=(1, 1)), dtype=torch.float)

    num_runs = 50
    steps = (2, 3, 4, 5, 6)
    averages = []
    for num in steps:
        print('testing %i...' % num)
        total_epochs = 0
        for _ in range(num_runs):
            epochs = count_epochs_to_converge(pagnn, X, T, num)
            total_epochs += epochs
        avg = total_epochs / num_runs
        averages.append(avg)
        print('num_steps=%i average epochs to converge: %.2f' % (num, avg))

    print(dict(zip(steps, averages)))

