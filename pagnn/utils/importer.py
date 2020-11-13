"""This file is responsible for importing various networks into the PAGNN architecture"""

import torch
from pagnn import PAGNNLayer



def count_neurons(net, return_layers=False):
    first = None
    extra = 0
    last = 0
    layers = 0
    for child in net.children():
        if type(child) == torch.nn.Linear:
            layers += 1
            in_neurons = child.in_features
            if first is None:
                first = in_neurons
            else:
                extra += in_neurons 
            last = child.out_features

    if return_layers:
        return first, extra, last, layers
    return first, extra, last


def import_ffnn(ffnn, activation):
    """Ingest a FFNN into the PAGNN architecture"""

    first, extra, last, layers = count_neurons(ffnn, return_layers=True)
    pagnn = PAGNNLayer(first, last, extra, steps=layers, activation=activation, retain_state=False)

    return pagnn
