import torch
import torch.nn.functional as F

import pandas as pd

from PAGNN.pagnn import PAGNN
from PAGNN.utils.comparisons import FFNN, one_hot, separate_targets, get_train_and_test, get_dataloaders, compare

import networkx as nx


if __name__ == '__main__':
    df = pd.read_csv('datasets/mushroom_classification/mushrooms.csv')

    # one-hot encodings
    df = one_hot(df, 'cap-shape')
    df = one_hot(df, 'cap-surface')
    df = one_hot(df, 'cap-color')
    df = one_hot(df, 'bruises')
    df = one_hot(df, 'odor')
    df = one_hot(df, 'gill-attachment')
    df = one_hot(df, 'gill-spacing')
    df = one_hot(df, 'gill-size')
    df = one_hot(df, 'gill-color')
    df = one_hot(df, 'stalk-shape')
    df = one_hot(df, 'stalk-root')
    df = one_hot(df, 'stalk-surface-above-ring')
    df = one_hot(df, 'stalk-surface-below-ring')
    df = one_hot(df, 'stalk-color-above-ring')
    df = one_hot(df, 'stalk-color-below-ring')
    df = one_hot(df, 'veil-type')
    df = one_hot(df, 'veil-color')
    df = one_hot(df, 'ring-number')
    df = one_hot(df, 'ring-type')
    df = one_hot(df, 'spore-print-color')
    df = one_hot(df, 'population')
    df = one_hot(df, 'habitat')
    df = one_hot(df, 'class')

    df, targets = separate_targets(df, 'class')
    train_perc = 0.8
    data_tensors = get_train_and_test(df, targets, train_perc)
    (train_X, train_T), (test_X, test_T) = data_tensors

    batch_size = 1
    train_dl, test_dl = get_dataloaders(data_tensors, batch_size)

    D = train_X.shape[1]
    C = 2

    print('input features:', D, 'num classes:', C)

    pagnn_lr = 0.001
    extra_neurons = 5
    pagnn_model = PAGNN(D + C + extra_neurons, D, C, graph_generator=nx.generators.classic.complete_graph)
    pagnn = {
        'name': 'PAGNN',
        'model': pagnn_model,
        'num_steps': 5,
        'optimizer': torch.optim.Adam(pagnn_model.parameters(), lr=pagnn_lr),
    }

    ffnn_lr = 0.0001
    ffnn_model = FFNN(D, 25, C)
    ffnn = {
        'name': 'FFNN',
        'model': ffnn_model,
        'optimizer': torch.optim.Adam(ffnn_model.parameters(), lr=ffnn_lr),
    }

    criterion = F.cross_entropy
    epochs = 1
    compare((pagnn, ffnn), train_dl, test_dl, epochs, criterion)
