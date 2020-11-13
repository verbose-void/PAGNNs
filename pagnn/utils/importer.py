"""This file is responsible for importing various networks into the PAGNN architecture"""

import torch
from pagnn import PAGNNLayer


def get_linear_layers(net):
    linear_layers = []
    for child in net.children():
        if type(child) == torch.nn.Linear:
            linear_layers.append(child)
    return linear_layers


def count_neurons(net, return_layers=False):
    first = None
    extra = 0
    last = 0
    layers = 0
    for layer in get_linear_layers(net):
        layers += 1
        in_neurons = layer.in_features
        if first is None:
            first = in_neurons
        else:
            extra += in_neurons 
        last = layer.out_features

    if return_layers:
        return first, extra, last, layers
    return first, extra, last


def import_ffnn(ffnn, activation):
    """Ingest a FFNN into the PAGNN architecture"""

    # create equivalent adjacency structure
    first, extra, last, layers = count_neurons(ffnn, return_layers=True)
    pagnn = PAGNNLayer(first, last, extra, steps=layers, activation=activation, retain_state=False)

    # import synaptic weightings
    pagnn.zero_params()
    for i, layer in enumerate(get_linear_layers(ffnn)):
        print(pagnn.weight.shape, layer.weight.shape)
        pagnn.weight.data[0, 1] = layer.weight.data
        print(pagnn.bias.shape, layer.bias.shape)
        pagnn.bias.data[1] = layer.bias.data

    return pagnn
