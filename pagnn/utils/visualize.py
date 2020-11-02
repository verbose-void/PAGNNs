import networkx as nx
import numpy as np


def get_networkx_graph(pagnn, return_color_map=True):
    W = pagnn.weight.cpu().detach().numpy()
    G = nx.DiGraph(W)

    if not return_color_map:
        return G

    color_map = []
    for i, neuron in enumerate(W):
        # "input" neurons
        if i < pagnn._input_neurons:
            color_map.append('green')

        # "output" neurons
        elif i >= (pagnn._total_neurons - pagnn._output_neurons):
            color_map.append('blue')

        # "hidden" neurons
        else:
            color_map.append('gray')

    return G, color_map


def draw_networkx_graph(pagnn, mode='default'):
    G, color_map = get_networkx_graph(pagnn, return_color_map=True)

    if mode == 'default':
        nx.draw(G, with_labels=True, node_color=color_map)
    elif mode == 'ego':
        node_and_degree = G.degree()
        (largest_hub, degree) = sorted(node_and_degree, key=itemgetter(1))[-1]
        hub_ego = nx.ego_graph(G, largest_hub)
        pos = nx.spring_layout(hub_ego)
        nx.draw(hub_ego, pos, node_color='b', node_size=50, with_labels=False)
        nx.draw_networkx_nodes(hub_ego, pos, nodelist=[largest_hub], node_size=300, node_color='r')
    elif mode == 'karate_club':
        nx.draw_circular(G, with_labels=True, node_color=color_map)
    elif mode == 'roget':
        UG = G.to_undirected()
        nx.draw_circular(UG, node_color=color_map, node_size=1, line_color='grey', linewidths=0, width=0.1)
    elif mode == 'football':
        nx.draw(G, node_color=color_map, node_size=50, line_color='grey', linewidths=0, width=0.1)
    elif mode == 'scaled_weights':
        degrees = np.array([G.degree[n] for n in range(pagnn._total_neurons)])
        degrees = degrees - np.min(degrees)
        degrees = degrees / np.max(degrees)
        degrees *= 200
        degrees += 10
        weightings = np.abs(pagnn.weight.cpu().detach().numpy().flatten())
        weightings = weightings - np.min(weightings)
        weightings = weightings / np.max(weightings)
        weightings *= 1
        weightings += 0.1
        nx.draw(G, node_color=color_map, node_size=degrees, width=weightings)
    else:
        raise Exception('mode %s not found.' % mode)
