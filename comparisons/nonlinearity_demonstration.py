import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from PAGNN.pagnn import PAGNN
from PAGNN.utils.comparisons import FFNN


def train(model, X, T, epochs=1, num_steps=1, lr=0.001, ffnn=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    model.train()
    with torch.enable_grad():
        best_loss = float('inf')
        best_weights = None

        for epoch in range(epochs):
            avg_loss = 0
            for x, t in zip(X, T):
                optimizer.zero_grad()

                t = t.unsqueeze(0)
                if ffnn:
                    y = model(x.unsqueeze(0))
                else:
                    y = model(x.unsqueeze(0), num_steps=num_steps)
                loss = F.mse_loss(y, t)
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()

            
            avg_loss = avg_loss / len(X)
            if ffnn:
                print('[ffnn] epoch %i loss:' % epoch, avg_loss)
            else:
                print('epoch %i loss:' % epoch, avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_weights = model.state_dict()

            history.append(avg_loss)

    print('best loss:', best_loss)
    model.load_state_dict(best_weights)
    return history


if __name__ == '__main__':
    seed = 666
    np.random.seed(seed)
    torch.manual_seed(seed)

    N = 40
    x = np.linspace(0, 2, N)

    m = 1.15
    b = 0
    linear = {
        'x': x,
        'y': m * x + b # linear data
    }
    linear['y'] += np.random.uniform(size=N, low=-1, high=1)

    non_linear = {
        'x': x,
        'y': F.tanh(torch.tensor(4 - x * 4)) - 1
    }
    non_linear['y'] += np.random.uniform(size=N, low=-0.1, high=0.1)

    # pagnn = PAGNN(2, 1, 1)
    # num_steps = 1

    # untrained_y = []
    # for x_sample in x:
    #     x_sample = torch.tensor([[x_sample]]).float()
    #     y = pagnn(x_sample, num_steps=num_steps)
    #     untrained_y.append(y.item())

    X = torch.tensor(linear['x'], dtype=torch.float).unsqueeze(-1)

    # set up plots
    fig, axs = plt.subplots(3, 2, figsize=(10, 10))

    """ LINEAR """
    pagnn = PAGNN(2, 1, 1)
    num_steps = 1
    T = torch.tensor(linear['y'], dtype=torch.float)
    untrained_y = pagnn(X, num_steps=num_steps).detach()
    pagnn_history = train(pagnn, X, T, epochs=50, num_steps=num_steps, lr=0.01)
    trained_y = pagnn(X, num_steps=num_steps).detach()
    ax = axs[0, 0]
    ax.set_title('Linear Data')
    ax.scatter(x, linear['y'], marker='o', label='X')
    # ax.plot(x, untrained_y, marker='o', label='Untrained PAGNN')
    ax.plot(x, trained_y, marker='o', label='Trained PAGNN')
    axs[2, 0].set_title('training loss')
    axs[2, 0].plot(pagnn_history, label='PAGNN')
    ax.legend()
    pagnn.draw_networkx_graph(ax=axs[1, 0])

    """ NON-LINEAR """
    ffnn = FFNN(1, 3, 1)
    pagnn = PAGNN(8, 1, 1, initial_sparsity=0)
    num_steps = 5
    T = torch.tensor(non_linear['y'], dtype=torch.float)
    untrained_y = pagnn(X, num_steps=num_steps).detach()

    pagnn_history = train(pagnn, X, T, epochs=2000, num_steps=num_steps, lr=0.0001)
    ffnn_history = train(ffnn, X, T, epochs=2000, lr=0.01, ffnn=True)

    trained_y = pagnn(X, num_steps=num_steps).detach()
    trained_ffnn_y = ffnn(X).detach()
    ax = axs[0, 1]
    ax.set_title('Non-Linear Data')
    ax.scatter(x, non_linear['y'], marker='o', label='X')
    # ax.plot(x, untrained_y, marker='o', label='Untrained PAGNN')
    ax.plot(x, trained_y, marker='o', label='Trained PAGNN')
    ax.plot(x, trained_ffnn_y, marker='o', label='Trained FFNN')
    axs[2, 1].set_title('training loss')
    axs[2, 1].plot(pagnn_history, label='PAGNN')
    axs[2, 1].plot(ffnn_history, label='FFNN')
    ax.legend()
    pagnn.draw_networkx_graph(ax=axs[1, 1])
    print('energy scalar', pagnn.structure_adj_matrix.energy_scalar)

    plt.savefig('figures/nonlinearity_demonstration.png', transparent=True)
    plt.show()
