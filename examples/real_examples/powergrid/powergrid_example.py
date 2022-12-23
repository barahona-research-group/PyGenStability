import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pygenstability import run, plotting


edges    = np.genfromtxt('UCTE_edges.txt')
location = np.genfromtxt('UCTE_nodes.txt')
posx = location[:,1]
posy = location[:,2]
pos  = {}

edges = np.array(edges,dtype=np.int32)
G = nx.Graph() #empty graph
G.add_edges_from(edges) #add edges

# resetting label ids
G = nx.convert_node_labels_to_integers(G, label_attribute = 'old_label' )

# updating label names and applying positions as attribute
for i in G.nodes():
    pos[i] = np.array([posx[G.nodes[i]['old_label']-1], posy[G.nodes[i]['old_label']-1]])
    G.nodes[i]['pos'] = pos[i].reshape(-1)

adjacency = nx.adjacency_matrix(G)

all_results = run(adjacency, min_scale=-1.5, max_scale = 2, 
                  n_scale = 30, constructor='linearized')

_ = plotting.plot_scan(all_results, use_plotly=True)

_ = plotting.plot_scan(all_results, use_plotly=False)

plotting.plot_optimal_partitions(G, all_results, figsize=(12,10))





