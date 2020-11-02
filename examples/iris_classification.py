import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from pagnn.utils.comparisons import FFNN, one_hot, normalize_inplace, count_params, compare
from pagnn.utils.visualize import draw_networkx_graph
from pagnn.pagnn import PAGNNLayer

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

    # split dataset into train & test
    X = torch.tensor(df.to_numpy()).float()
    T = torch.tensor(targets.to_numpy()).float()
    T = torch.argmax(T, dim=1)
    train_perc = 0.67
    split = int(train_perc * X.shape[0])
    train_X, test_X = X[:split], X[split:]
    train_T, test_T = T[:split], T[split:]

    # create data loaders
    batch_size = 10
    train_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_X, train_T), batch_size=batch_size)
    test_dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_X, test_T), batch_size=batch_size)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_dicts = []
    configs = [
        {'num_steps': 1, 'activation': None},
        {'num_steps': 2, 'activation': None},
        {'num_steps': 3, 'activation': None},
        # {'num_steps': 1, 'activation': 'relu'}, # doesn't work
        {'num_steps': 2, 'activation': 'relu'},
        {'num_steps': 3, 'activation': 'relu'},
    ]

    for config in configs:
        pagnn_lr = 0.01
        extra_neurons = 0
        n = D + C + extra_neurons
        activation_func = None
        if config['activation'] == 'relu':
            activation_func = F.relu
        pagnn_model = PAGNNLayer(D, C, extra_neurons, steps=config['num_steps'], retain_state=False, activation=activation_func) # graph_generator=nx.generators.classic.complete_graph)
        pagnn_model.to(device)
        if config['activation'] is None:
            model_name = 'PAGNNLayer(n=%i, #p=%i, num_steps=%i)' % (n, count_params(pagnn_model), config['num_steps'])
        else:
            model_name = 'PAGNNLayer(n=%i, #p=%i, num_steps=%i) + %s' % (n, count_params(pagnn_model), config['num_steps'], str(config['activation']))
        pagnn = {
            'name': model_name,
            'model': pagnn_model,
            # 'num_steps': config['num_steps'],
            'optimizer': torch.optim.Adam(pagnn_model.parameters(), lr=pagnn_lr),
        }

        model_dicts.append(pagnn)

    ffnn_lr = 0.01
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
    fig.suptitle('Iris Classification - (PAGNN vs FFNN)', fontsize=24)

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

    plt.savefig('examples/figures/iris_classification.png', transparent=True)
    plt.show()
