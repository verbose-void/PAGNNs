import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pagnn.pagnn import PAGNNLayer
from pagnn.utils.comparisons import FFNN, one_hot, separate_targets, get_train_and_test, get_dataloaders, compare, count_params
from pagnn.utils.visualize import draw_networkx_graph


if __name__ == '__main__':
    seed = 666
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    df = pd.read_csv('datasets/mushrooms.csv')

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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_dicts = []

    linear_lr = 0.01
    linear_model = torch.nn.Linear(D, C)
    linear_model.to(device)
    linear = {
        'name': 'Linear(%i, %i, #p=%i)' % (D, C, count_params(linear_model)),
        'model': linear_model,
        'optimizer': torch.optim.Adam(linear_model.parameters(), lr=linear_lr),
    }

    model_dicts.append(linear)

    configs = [
        {'num_steps': 1},
        {'num_steps': 3},
    ]

    for config in configs:
        pagnn_lr = 0.001
        extra_neurons = 0
        n = D + C + extra_neurons
        pagnn_model = PAGNNLayer(D, C, extra_neurons, steps=config['num_steps'], retain_state=False) # graph_generator=nx.generators.classic.complete_graph)
        pagnn_model.to(device)
        pagnn = {
            'name': 'PAGNNLayer(n=%i, #p=%i, steps=%i)' % (n, count_params(pagnn_model), config['num_steps']),
            'model': pagnn_model,
            # 'num_steps': config['num_steps'],
            'optimizer': torch.optim.Adam(pagnn_model.parameters(), lr=pagnn_lr),
        }

        model_dicts.append(pagnn)

    ffnn_lr = 0.0001
    ffnn_model = FFNN(D, D, C)
    ffnn_model.to(device)
    ffnn = {
        'name': 'FFNN(%i, %i, %i, #p=%i)' % (D, D, C, count_params(ffnn_model)),
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
    draw_networkx_graph(pagnn['model'], mode='scaled_weights')
    plt.title('%s architecture' % pagnn['name'])

    plt.savefig('examples/figures/mushroom_classification.png', transparent=True)
    plt.show()
