import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from PAGNN.pagnn import PAGNN

import numpy as np

from math import isnan

import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx


if __name__ == '__main__':
    seed = 666
    if seed is not None:
        torch.manual_seed(seed)

    pagnn = PAGNN(5, 1, 1, initial_sparsity=0.5)
    print(pagnn)

    linear_model = torch.nn.Linear(1, 1)
    print(linear_model)

    # load data into torch tensors
    train_df = pd.read_csv('datasets/linear_regression/train.csv')
    train_df = train_df.dropna()
    test_df = pd.read_csv('datasets/linear_regression/test.csv')
    test_df = test_df.dropna()
    train_x, train_y = torch.tensor(train_df['x'].to_numpy()), torch.tensor(train_df['y'].to_numpy())
    test_x, test_y = torch.tensor(test_df['x'].to_numpy()), torch.tensor(test_df['y'].to_numpy())
    # print('train shapes', train_x.shape, train_y.shape)
    # print('test shapes', test_x.shape, test_y.shape)

    # create data loaders
    batch_size = 1
    train_dl = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size)
    test_dl = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)

    pagnn_lr = 0.001
    baseline_lr = 0.0001
    optimizer = torch.optim.Adam(pagnn.parameters(), lr=pagnn_lr)
    baseline_optimizer = torch.optim.Adam(linear_model.parameters(), lr=baseline_lr)
    num_steps = 5

    pagnn_history = {'train_loss': [], 'test_loss': []}
    baseline_history = {'train_loss': [], 'test_loss': []}
    
    for epoch in range(10):
        with torch.enable_grad():
            pagnn_total_loss = 0
            baseline_total_loss = 0

            for x, t in train_dl:
                optimizer.zero_grad()

                x = x.float().unsqueeze(-1)
                t = t.float().unsqueeze(-1)

                y = pagnn(x, num_steps=num_steps).unsqueeze(-1)
                baseline_y = linear_model(x)

                loss = F.mse_loss(y, t)
                baseline_loss = F.mse_loss(baseline_y, t)
                pagnn_total_loss += loss.item()
                baseline_total_loss += baseline_loss.item()

                loss.backward()
                optimizer.step()

                baseline_loss.backward()
                baseline_optimizer.step()

            pagnn_avg_loss = pagnn_total_loss / len(train_dl)
            baseline_avg_loss = baseline_total_loss / len(train_dl)
            print('[PAGNN] average loss for epoch %i: %f' % (epoch, pagnn_avg_loss))
            print('[BASELINE] average loss for epoch %i: %f' % (epoch, baseline_avg_loss))

            pagnn_history['train_loss'].append(pagnn_avg_loss)
            baseline_history['train_loss'].append(baseline_avg_loss)

        with torch.no_grad():
            total_loss = 0
            baseline_total_loss = 0
            for x, t in test_dl:
                x = x.float().unsqueeze(-1)
                t = t.float().unsqueeze(-1)

                y = pagnn(x, num_steps=num_steps).unsqueeze(-1)
                baseline_y = linear_model(x)

                loss = F.mse_loss(y, t)
                total_loss += loss.item()

                baseline_loss = F.mse_loss(baseline_y, t)
                baseline_total_loss += baseline_loss.item()

            pagnn_avg_loss = total_loss / len(test_dl)
            baseline_avg_loss = baseline_total_loss / len(test_dl)

            print('[PAGNN] testing loss:', pagnn_avg_loss)
            print('[BASELINE] testing loss:', baseline_avg_loss)

            pagnn_history['test_loss'].append(pagnn_avg_loss)
            baseline_history['test_loss'].append(baseline_avg_loss)

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('Linear Regression - (PAGNN vs torch.nn.Linear)', fontsize=24)

    plt.subplot(222)
    plt.plot(pagnn_history['train_loss'], label='PAGNN (lr: %f)' % pagnn_lr)
    plt.plot(baseline_history['train_loss'], label='Baseline (lr: %f)' % baseline_lr)
    plt.legend()
    plt.title('train loss')

    plt.subplot(221)
    plt.plot(pagnn_history['test_loss'], label='PAGNN')
    plt.plot(baseline_history['test_loss'], label='Baseline')
    plt.legend()
    plt.title('test loss')

    plt.subplot(212)
    G = nx.Graph(pagnn.structure_adj_matrix.weight.detach().numpy())
    nx.draw(G, with_labels=True)
    plt.title('PAGNN architecture')

    plt.savefig('figures/linear_regression.png', transparent=True)
    plt.show()
