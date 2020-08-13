import torch
import torch.nn.functional as F

import pandas as pd

from PAGNN.pagnn import PAGNN
from PAGNN.utils.comparisons import FFNN, one_hot, separate_targets, get_train_and_test, get_dataloaders, compare

import networkx as nx

import matplotlib.pyplot as plt

import numpy as np


if __name__ == '__main__':
    seed = 666
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

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
    data_tensors = get_train_and_test(df, targets, train_perc, dtype=torch.float)
    (train_X, train_T), (test_X, test_T) = data_tensors

    batch_size = 10
    train_dl, test_dl = get_dataloaders(data_tensors, batch_size)

    D = train_X.shape[1]
    C = 2

    print('input features:', D, 'num classes:', C)

    device = torch.device('cuda') # CUDA for this test is faster (quite a bit more neurons)

    model_dicts = []
    configs = [
        {'initial_sparsity': 0, 'num_steps': 5},
        {'initial_sparsity': 0.1, 'num_steps': 5},
        {'initial_sparsity': 0.5, 'num_steps': 5},
        {'initial_sparsity': 0.8, 'num_steps': 5},
        {'initial_sparsity': 0.9, 'num_steps': 5},
        {'initial_sparsity': 0.98, 'num_steps': 5},
    ]
    
    for config in configs:
        pagnn_lr = 0.001
        extra_neurons = 5
        n = D + C + extra_neurons
        pagnn_model = PAGNN(n, D, C, initial_sparsity=config['initial_sparsity']) # graph_generator=nx.generators.classic.complete_graph)
        pagnn_model.to(device)
        pagnn = {
            'name': 'PAGNN(neurons=%i, initial_sparsity=%f)' % (n, config['initial_sparsity']),
            'model': pagnn_model,
            'num_steps': config['num_steps'],
            'optimizer': torch.optim.Adam(pagnn_model.parameters(), lr=pagnn_lr),
        }

        model_dicts.append(pagnn)

    ffnn_lr = 0.0001
    ffnn_model = FFNN(D, D, C)
    ffnn_model.to(device)
    ffnn = {
        'name': 'FFNN(%i, %i, %i)' % (D, D, C),
        'model': ffnn_model,
        'optimizer': torch.optim.Adam(ffnn_model.parameters(), lr=ffnn_lr),
    }

    model_dicts.append(ffnn)

    # print('pagnn num params:', sum(p.numel() for p in pagnn_model.parameters()))
    # print('ffnn num params:', sum(p.numel() for p in ffnn_model.parameters()))

    criterion = F.cross_entropy
    epochs = 25
    compare(model_dicts, train_dl, test_dl, epochs, criterion, test_accuracy=True, device=device)
    
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('Mushroom Classification - (PAGNN vs FFNN)', fontsize=24)

    plt.subplot(221)
    for model_dict in model_dicts:
        plt.plot(model_dict['train_history'], label=model_dict['name'])
    # plt.plot(ffnn['train_history'], label='FFNN (lr: %f)' % ffnn_lr)
    plt.legend()
    plt.title('train loss')

    plt.subplot(222)
    for model_dict in model_dicts:
        plt.plot(model_dict['test_history'], label=model_dict['name'])
    # plt.plot(ffnn['test_history'], label='FFNN')
    plt.legend()
    plt.title('test accuracy')

    plt.subplot(212)
    print('creating graph...')
    pagnn = model_dicts[-2] # get the last pagnn network defined
    pagnn['model'].draw_networkx_graph(mode='scaled_weights')
    plt.title('%s architecture' % pagnn['name'])

    plt.savefig('figures/mushroom_classification.png', transparent=True)
    plt.show()
