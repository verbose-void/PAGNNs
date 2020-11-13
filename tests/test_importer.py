import torch
import numpy as np

from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F

from pagnn.utils.importer import import_ffnn, count_neurons


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

np.random.seed(666)
torch.manual_seed(666)


def test_count_neurons():
    net = Sequential(
        Linear(17, 3),
        ReLU(),
        Linear(3, 19),
        ReLU(),
        Linear(19, 1)
    )

    first, extra, last = count_neurons(net)
    assert first == 17
    assert extra == 22
    assert last == 1


def test_import_ffnn():
    X = torch.rand((50, 1))
    T = torch.rand((50, 1))

    print('input data:')
    print(X)

    """
    net1 = Sequential(
        Linear(1, 5),
        ReLU(),
        Linear(5, 5,),
        ReLU(),
        Linear(5, 1)
    )
    """
    net1 = Sequential(
        Linear(1, 1),
        ReLU(),
    )

    print('FFNN:')
    print(net1)

    Y = net1(X)

    print('FFNN output:')
    print(Y)

    net1_pagnn = import_ffnn(net1, F.relu)

    print('imported PAGNN:')
    print(net1_pagnn)

    imported_Y = net1_pagnn(X)

    print('imported PAGNN output:')
    print(imported_Y)

    print('FFNN out shape:', Y.shape, 'PAGNN out shape:', imported_Y.shape)

    error = F.mse_loss(Y, imported_Y)
    
    assert error < 1e-3, 'All elements of Y must equal imported_Y. MSError: %f' % error
