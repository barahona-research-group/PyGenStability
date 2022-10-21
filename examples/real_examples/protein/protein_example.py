import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pygenstability import run, plotting
import os

pdb_id = '2RH5'
G = nx.read_gpickle(os.path.join('data', pdb_id + ".gpickle"))
adjacency = nx.adjacency_matrix(G)

all_results = run(adjacency, min_scale=-1.5, max_scale = 2, 
                  n_scale = 30, constructor='linearized')

_ = plotting.plot_scan(all_results, use_plotly=True)

_ = plotting.plot_scan(all_results, use_plotly=False)

