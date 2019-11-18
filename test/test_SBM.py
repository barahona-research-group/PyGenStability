import os as os
import numpy as np
import networkx as nx
import pylab as plt
import networkx as nx

import pygenstability.pygenstability as pgs

#set a simple SBM model
sizes = [10, 15, 9]
probs = [[0.7, 0.05, 0.08],   
         [0.05, 0.8, 0.02],   
         [0.08, 0.02, 0.80]]

G = nx.stochastic_block_model(sizes, probs, seed=0)

#need to set the weights to 1
for i,j in G.edges():
    G[i][j]['weight'] = 1

#ground truth
community_labels = [G.nodes[i]['block'] for i in G]

#spring layout
pos = nx.spring_layout(G,weight=None,scale=1)

#draw the graph with ground truth 
plt.figure(figsize=(5,4))
nx.draw(G,pos=pos,node_color=community_labels)
plt.title("Ground truth communities")
plt.savefig('ground_truth.png', bbox_inches='tight')

#number of Louvain runs for each time
louvain_runs = 50

# possible choices of type of clustering:
# continuous_combinatorial: continuous RW with combintaorial Laplacian
# continuous_normalized:    continuous RW with normalized Laplacian
# linearized:               modularity with time parameter
# modularity_signed:       modularity of signed networks, (Arenas et al. 2008)
stability_type = 'continuous_normalized'

#crete the stability object
stability = pgs.PyGenStability(G, stability_type, louvain_runs) 

#to compute MI between al Louvain, or the top n_mi
stability.all_mi = False 
#if all_mi = False, number of top Louvain run to use for MI
stability.n_mi = 10  

#number of cpu for parallel compuations
stability.n_processes_louv = 1 # of Louvain
stability.n_processes_mi   = 1 # of MI

# apply the postprocessing or not
stability.post_process = False
stability.n_neigh = 10 #use only these number of neighbors for postprocessing

#run a single time, and print result
stability.run_single_stability(time = 1.)
stability.print_single_result(1, 1)

#scan over a time interval
times = np.logspace(-0.5, 0.5, 100)
stability.scan_stability(times, disp=False)

#stability.plot_scan()
#plt.savefig('scan_results.svg', bbox_inches='tight')

#now plot the community structures at each time in a folder
def plot_communities(t):
    #plot the communities on the network
    node_color = stability.stability_results.iloc[t]['community_id']
    plt.figure()

    nx.draw_networkx_nodes(G, pos=pos,node_color=node_color, node_size = 100, cmap=plt.get_cmap('tab20'))
    nx.draw_networkx_edges(G, pos=pos,width = 0.5, edge_color='0.5')
    plt.axis('off')
    plt.title(str(r'$log_{10}(time) =$ ')+ str(np.round(np.log10(times[t]),2)) +', with ' + str(stability.stability_results.iloc[t]['number_of_communities'])+' communities')
    plt.savefig('communities/time_'+str(t)+'.svg', bbox_inches='tight')
    plt.close()

#create a subfolder to store images
if not os.path.isdir('communities'):
    os.mkdir('communities')

#plot the communities at each Markov time
#for t in range(len(times)):
#    plot_communities(t)

#plt.show()
