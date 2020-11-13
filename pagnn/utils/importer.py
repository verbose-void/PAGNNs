"""This file is responsible for importing various networks into the PAGNN architecture"""

import torch
from pagnn import PAGNNLayer



def count_neurons(net):
    first = None
    extra = 0
    last = 0
    for child in net.children():
        if type(child) == torch.nn.Linear:
            in_neurons = child.in_features
            if first is None:
                first = in_neurons
            else:
                extra += in_neurons 
            last = child.out_features
    return first, extra, last


def import_ffnn(ffnn):
    """Ingest a FFNN into the PAGNN architecture"""

    first, extra, last = count_neurons(ffnn)
    print(first, extra, last)

    return None
