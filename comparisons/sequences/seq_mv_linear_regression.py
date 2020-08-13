import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from PAGNN.pagnn import PAGNN

import numpy as np

from math import isnan

import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx


def one_hot(df, key):
    return pd.concat([df, pd.get_dummies(df[key], prefix=key)], axis=1).drop(key, axis=1)


def normalize_inplace(df, key):
    df[key] = df[key] / np.linalg.norm(df[key])


if __name__ == '__main__':
    seed = 666
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # load data
    df = pd.read_csv('datasets/mv_linear_regression/insurance.csv').dropna()

    # normalize data
    normalize_inplace(df, 'age')
    normalize_inplace(df, 'bmi')
    normalize_inplace(df, 'children')
    normalize_inplace(df, 'charges')

    # separate targets
    targets = df['charges']
    df = df.drop('charges', axis=1)

    # one-hot encodings
    df = one_hot(df, 'sex')
    df = one_hot(df, 'smoker')
    df = one_hot(df, 'region')

    D = len(df.columns)
    print('number of data features:', D)

    # create models
    pagnn = PAGNN(D + 5, 1, 1, initial_sparsity=0.8) # graph_generator=nx.generators.classic.complete_graph)
    print(pagnn)
    linear_model = torch.nn.Linear(D, 1)
    print(linear_model)

    # split dataset into train & test
    X = torch.tensor(df.to_numpy())
    T = torch.tensor(targets.to_numpy())
    train_perc = 0.8
    split = int(train_perc * X.shape[0])
    train_X, test_X = X[:split], X[split:]
    train_T, test_T = T[:split], T[split:]

    # create data loaders
    batch_size = 10
    train_dl = DataLoader(TensorDataset(train_X, train_T), batch_size=batch_size)
    test_dl = DataLoader(TensorDataset(test_X, test_T), batch_size=batch_size)

    pagnn_lr = 0.001
    baseline_lr = 0.001
    optimizer = torch.optim.Adam(pagnn.parameters(), lr=pagnn_lr)
    baseline_optimizer = torch.optim.Adam(linear_model.parameters(), lr=baseline_lr)
    num_steps = 1

    pagnn_history = {'train_loss': [], 'test_loss': []}
    baseline_history = {'train_loss': [], 'test_loss': []}

    # NOTE: using CPU (faster for this specific test, not very many neurons)
    
    for epoch in range(20):
        with torch.enable_grad():
            pagnn_total_loss = 0
            baseline_total_loss = 0

            for x, t in train_dl:
                x = x.float()
                t = t.float().unsqueeze(-1)

                optimizer.zero_grad()
                baseline_optimizer.zero_grad()

                y = pagnn(x.unsqueeze(-1), use_sequence=True)
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
                x = x.float()
                t = t.float().unsqueeze(-1)

                y = pagnn(x.unsqueeze(-1), num_steps=num_steps, use_sequence=True)
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
    fig.suptitle('Sequence-Fed Multi-Variate Linear Regression - (PAGNN vs torch.nn.Linear)', fontsize=24)

    plt.subplot(221)
    plt.plot(pagnn_history['train_loss'], label='PAGNN (lr: %f)' % pagnn_lr)
    plt.plot(baseline_history['train_loss'], label='Baseline (lr: %f)' % baseline_lr)
    plt.legend()
    plt.title('train loss')

    plt.subplot(222)
    plt.plot(pagnn_history['test_loss'], label='PAGNN')
    plt.plot(baseline_history['test_loss'], label='Baseline')
    plt.legend()
    plt.title('test loss')

    plt.subplot(212)
    G, color_map = pagnn.get_networkx_graph(return_color_map=True)
    nx.draw(G, with_labels=True, node_color=color_map)
    plt.title('PAGNN architecture')

    plt.savefig('figures/seq_mv_linear_regression.png', transparent=True)
    plt.show()
