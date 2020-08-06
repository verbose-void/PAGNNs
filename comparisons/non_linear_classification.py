import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from PAGNN.pagnn import PAGNN

import numpy as np

from math import isnan

import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx

from sklearn.utils import shuffle


class FFNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.activation = torch.sigmoid
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


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
    df = pd.read_csv('datasets/non_linear_classification/iris.csv').dropna()
    df = shuffle(df)

    # normalize data
    # normalize_inplace(df, 'SepalLengthCm')
    # normalize_inplace(df, 'SepalWidthCm')
    # normalize_inplace(df, 'PetalLengthCm')
    # normalize_inplace(df, 'PetalWidthCm')

    # one-hot encodings
    df = one_hot(df, 'Species')

    # separate targets
    filter_col = [col for col in df if col.startswith('Species')]
    targets = df[filter_col]
    df = df.drop(filter_col, axis=1)
    df = df.drop('Id', axis=1)

    D = len(df.columns)
    C = len(targets.columns)
    print('number of data features:', D)
    print('number of classes:', C)

    # create models
    pagnn = PAGNN(D + C + 5, D, C, initial_sparsity=0.2)
    print(pagnn)
    linear_model = FFNN(D, 25, C) # torch.nn.Linear(D, C)
    print(linear_model)

    # split dataset into train & test
    X = torch.tensor(df.to_numpy())
    T = torch.tensor(targets.to_numpy())
    T = torch.argmax(T, dim=1)
    train_perc = 0.67
    split = int(train_perc * X.shape[0])
    train_X, test_X = X[:split], X[split:]
    train_T, test_T = T[:split], T[split:]

    # create data loaders
    batch_size = 1
    train_dl = DataLoader(TensorDataset(train_X, train_T), batch_size=batch_size)
    test_dl = DataLoader(TensorDataset(test_X, test_T), batch_size=batch_size)

    pagnn_lr = 0.0001
    baseline_lr = 0.00001
    optimizer = torch.optim.Adam(pagnn.parameters(), lr=pagnn_lr)
    baseline_optimizer = torch.optim.Adam(linear_model.parameters(), lr=baseline_lr)
    num_steps = 5

    pagnn_history = {'train_loss': [], 'test_accuracy': []}
    baseline_history = {'train_loss': [], 'test_accuracy': []}
    
    try:
        for epoch in range(1000):
            pagnn.train()
            linear_model.train()

            with torch.enable_grad():
                pagnn_total_loss = 0
                baseline_total_loss = 0

                for x, t in train_dl:
                    optimizer.zero_grad()

                    x = x.float()

                    y = pagnn(x, num_steps=num_steps).unsqueeze(0)
                    baseline_y = linear_model(x)

                    loss = F.cross_entropy(y, t)
                    baseline_loss = F.cross_entropy(baseline_y, t)
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

            pagnn.eval()
            linear_model.eval()
            with torch.no_grad():
                total_correct = 0.0
                baseline_total_correct = 0.0
                for x, t in test_dl:
                    x = x.float()

                    y = pagnn(x, num_steps=num_steps).unsqueeze(0)
                    baseline_y = linear_model(x)

                    pred = torch.argmax(y, axis=1)
                    baseline_pred = torch.argmax(baseline_y, axis=1)

                    total_correct += torch.sum(pred == t).item()
                    baseline_total_correct += torch.sum(baseline_pred == t).item()

                pagnn_accuracy = total_correct / len(test_dl.dataset)
                baseline_accuracy = baseline_total_correct / len(test_dl.dataset)

                print('[PAGNN] testing accuracy:', pagnn_accuracy)
                print('[BASELINE] testing accuracy:', baseline_accuracy)

                pagnn_history['test_accuracy'].append(pagnn_accuracy)
                baseline_history['test_accuracy'].append(baseline_accuracy)
    except KeyboardInterrupt:
        print('early exit keyboard interrupt')

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('Non-Linear Classification - (PAGNN vs FFNN(4, 25, 3))', fontsize=24)

    plt.subplot(221)
    plt.plot(pagnn_history['train_loss'], label='PAGNN (lr: %f)' % pagnn_lr)
    plt.plot(baseline_history['train_loss'], label='Baseline (lr: %f)' % baseline_lr)
    plt.legend()
    plt.title('train loss')

    plt.subplot(222)
    plt.plot(pagnn_history['test_accuracy'], label='PAGNN')
    plt.plot(baseline_history['test_accuracy'], label='Baseline')
    plt.legend()
    plt.title('test accuracy')

    plt.subplot(212)
    G, color_map = pagnn.get_networkx_graph(return_color_map=True)
    nx.draw(G, with_labels=True, node_color=color_map)
    plt.title('PAGNN architecture')

    plt.savefig('figures/non_linear_classification.png', transparent=True)
    plt.show()
