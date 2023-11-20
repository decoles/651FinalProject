import networkx as nx
import numpy as np
import torch
from torch_geometric.utils.convert import from_networkx


# Make the networkx graph
G = nx.Graph()

# Add some cars (just do 4 for now)
G.add_nodes_from([
      (1, {'y': 1, 'x': 0.5}),
      (2, {'y': 2, 'x': 0.2}),
      (3, {'y': 3, 'x': 0.3}),
      (4, {'y': 4, 'x': 0.1}),
      (5, {'y': 5, 'x': 0.2}),
])

# Add some edges
G.add_edges_from([
                  (1, 2), (1, 4), (1, 5),
                  (2, 3), (2, 4),
                  (3, 2), (3, 5),
                  (4, 1), (4, 2),
                  (5, 1), (5, 3)
])

# Convert the graph into PyTorch geometric
pyg_graph = from_networkx(G)

#print node attributes
print(pyg_graph.x)

nx.draw(G=G, pos=nx.spring_layout(G), with_labels=True)