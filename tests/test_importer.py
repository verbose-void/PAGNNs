import torch
import numpy as np

from torch.nn import Sequential, Linear, ReLU
import torch.nn.functional as F

from pagnn.pagnn import import_ffnn, count_neurons
from pagnn.p_resnet import p_resnet18
from torchvision.models import resnet18


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


def assert_import_equivalence(net, X):
    print('input data:')
    print(X)

    print('Input net:')
    print(net)

    Y = net(X)

    print('output:')
    print(Y)

    pagnn = import_ffnn(net, F.relu)

    print('imported PAGNN:')
    print(pagnn)

    imported_Y = pagnn(X)

    print('PAGNN output:')
    print(imported_Y)

    print('original out shape:', Y.shape, 'PAGNN out shape:', imported_Y.shape)

    error = F.mse_loss(Y, imported_Y)
    
    assert error < 1e-3, 'All elements of Y must equal imported_Y. MSError: %f' % error


def test_import_linear_regression():
    X = torch.rand((50, 1)) * 100

    net = Sequential(
        Linear(1, 1, bias=True),
    )

    assert_import_equivalence(net, X)

def test_import_2layer():
    X = torch.rand((50, 1)) * 100

    net = Sequential(
        Linear(1, 2, bias=True),
        ReLU(),
        Linear(2, 1, bias=True),
    )

    assert_import_equivalence(net, X)

def test_import_3layer():
    X = torch.rand((50, 1)) * 100

    net = Sequential(
        Linear(1, 10, bias=True),
        ReLU(),
        Linear(10, 15, bias=False), # this layer does not have a bias term, tests dynamic bias allocation
        ReLU(),
        Linear(15, 5, bias=True),
    )

    assert_import_equivalence(net, X)

def test_import_resnet():
    res = resnet18(pretrained=True)
    pres = p_resnet18()
    pres.load_state_dict(res.state_dict())

    X = torch.rand((5, 3, 224, 224))
    
    pres_Y = pres(X)
    res_Y = res(X)

    error = F.mse_loss(pres_Y, res_Y)
    assert error < 1e-3, 'All elements of P-ResNet Y must equal ResNet Y. MSError: %f' % error
