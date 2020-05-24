import plotly.graph_objects as go
import matplotlib.pyplot as plt
import networkx as nx
import pickle

fn = 'best.pkl'
with open(fn, 'rb') as f:
    best_nn, rstate = pickle.load(f)

best_nn.revert_to_initial()
W = best_nn.graph_weights

color_map = []
for i, neuron in enumerate(W):
    if i <= 1:
        color_map.append('green')
    elif i >= len(W) - 4:
        color_map.append('blue')
    else:
        color_map.append('black')

G = nx.Graph(W)
#G.add_edge(1, 2)
#G.add_edge(1, 3)
nx.draw(G, with_labels=True, node_color=color_map)
plt.show()
