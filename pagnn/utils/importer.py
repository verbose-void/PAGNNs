"""This file is responsible for importing various networks into the PAGNN architecture"""

import torch
from pagnn import PAGNNLayer



def count_neurons(net):
    total = 0
    last = 0
    for child in net.children():
        if type(child) == torch.nn.Linear:
            in_neurons = child.in_features
            last = child.out_features
            total += in_neurons 
    return total + last


def import_ffnn(ffnn):
    """Ingest a FFNN into the PAGNN architecture"""

    neurons = count_neurons(ffnn)
    print(neurons)

    return None
