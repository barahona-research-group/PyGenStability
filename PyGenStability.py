import sys as sys
import numpy as np
import scipy as sc
import pandas as pd

import pylab as plt
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score
from functools import partial
from multiprocessing import Pool
#from multiprocessing import current_process
import os 
from timeit import default_timer as timer
from tqdm import tqdm
from array import array
from Louvain_to_python import clq 
a = clq.VectorPartition

#import cppyy
#cppyy.include("cpp/louvain_to_python.h")
#from cppyy.gbl import run_louvain

class PyGenStability(object):
    """ 
    Main class
    """

# =============================================================================
# init parameters
# =============================================================================
    def __init__(self, G, tpe, louvains_runs, precision):
        
        self.G = G # graph
        self.A = nx.adjacency_matrix(self.G, weight='weight')
        self.n = len(G.nodes) # number of nodes 
        self.m = len(G.edges) # number of edges 

        self.tpe = tpe #type of stability, linear or Markov
        self.use_spectral_gap = False #if True, rescale the Markov time by the spectral gap 
        self.louvain_runs = louvains_runs #number of Louvain run 
        self.precision = precision #precision threshold for Markov stability

        self.calcMI = True #compute the Mutual info score 
        self.all_mi = False
        self.n_mi = 5

        self.post_process = True #use post-processing of the scan as default
        self.n_neigh = 5 #number of nearby times to use in postprocessing
        self.time_computation = False #display the computation times of exponential, Louvain and MI

        #set the number of cores for parallel computing
        if "OMP_NUM_THREADS" in os.environ:
            self.n_processes_louv = int(os.environ['OMP_NUM_THREADS']) #for the Louvain run
            self.n_processes_mi = int(os.environ['OMP_NUM_THREADS']) #for the MI run
        else:
            self.n_processes_louv = int(1)
            self.n_processes_mi = int(1)  

        #for temporary files from cpp louvain code
        if not os.path.isdir('data'):
            os.mkdir('data')
        else:
            import shutil
            shutil.rmtree('data')
            os.mkdir('data')

        if not os.path.isdir('model'):
            os.mkdir('model')
        else:
            import shutil
            shutil.rmtree('model')
            os.mkdir('model')

        self.cpp_folder = '/home/arnaudon/codes/PyGenStability'

# =============================================================================
# create the generalized Louvain models
# =============================================================================
    def set_null_model(self):
        """
        create the null models
        """

        if self.tpe == 'continuous_combinatorial':
            self.pi = np.ones(self.n)/self.n
            self.null_model = np.array([self.pi, self.pi])

        if self.tpe == 'continuous_normalized' or self.tpe == 'linearized':
            self.degree = np.array(self.A.sum(1)).flatten()
            self.pi = self.degree / self.degree.sum()
            self.null_model = np.array([self.pi, self.pi])

        if self.tpe == 'continuous_signed':
            self.pi = np.zeros(self.n) 
            self.null_model = np.array([self.pi, self.pi])

        if self.tpe == 'modularity_signed':
            A = self.A.toarray()
            A_plus = 0.5*(A + abs(A))
            A_neg = -0.5*(A - abs(A))
            deg_plus = np.array(A_plus.sum(1)).flatten()
            deg_neg = np.array(A_neg.sum(1)).flatten()

            self.deg_norm = deg_plus.sum()+deg_neg.sum()

            if deg_neg.sum() < 1e-10:
                deg_neg_norm = np.zeros(self.n)
                deg_neg = np.zeros(self.n)
            else:
                deg_neg_norm = deg_neg/deg_neg.sum()

            if deg_plus.sum() < 1e-10:
                deg_plus_norm = np.zeros(self.n)
                deg_plus = np.zeros(self.n)
            else:
                deg_plus_norm = deg_plus/deg_plus.sum()


            self.null_model = np.array([deg_plus, deg_plus_norm, deg_neg, -deg_neg_norm])/self.deg_norm
    

    def set_quality_matrix(self, time):
        """
        create the quality matrix 
        """

        if self.tpe == 'continuous_combinatorial':
            if nx.is_directed(self.G): #for directed graph
                print("Not implemented for directed graph, need latest networkx version")
            else:
                L = 1.*nx.laplacian_matrix(self.G)
                
            if self.use_spectral_gap:
                l_min = abs(sc.sparse.linalg.eigs(L, which='SM',k=2)[0][1])
            else:
                l_min = 1. 
                
            ex = sc.sparse.linalg.expm(-time/l_min * L.toarray())
            self.Q = sc.sparse.csc_matrix(np.dot(np.diag(self.pi), ex))

        if self.tpe == 'continuous_normalized':
            if nx.is_directed(self.G): #for directed graph
                L = nx.directed_laplacian_matrix(self.G, walk_type='pagerank', alpha=0.8)
            else:
                L = np.diag(1./self.degree).dot(nx.laplacian_matrix(self.G).toarray())
            
            if self.use_spectral_gap:
                l_min = abs(sc.sparse.linalg.eigs(L, which='SM',k=2)[0][1])
            else:
                l_min = 1.
            
            ex = sc.sparse.linalg.expm(-time/l_min * L)
            self.Q = sc.sparse.csc_matrix(np.dot(np.diag(self.pi), ex))

        if self.tpe == 'linearized':
            self.Q = time*self.A/self.degree.sum()
        
        if self.tpe == 'continuous_signed':
            A = nx.adjacency_matrix(self.G).toarray()
            degree = abs(A).sum(1)
            D = np.diag(degree)
            L = sc.sparse.csr_matrix(D - A)
            
            if self.use_spectral_gap:
                l_min = abs(sc.sparse.linalg.eigs(L, which='SM',k=2)[0][1])
            else:
                l_min = 1.

            ex = sc.sparse.linalg.expm(-time/l_min * L.toarray())
            self.Q = sc.sparse.csc_matrix(ex)

        if self.tpe == 'modularity_signed':
            self.Q = time*self.A/self.deg_norm

        self.Q = (np.max(self.Q)*self.precision)*np.round(self.Q/((np.max(self.Q)*self.precision)))
        self.Q = sc.sparse.csc_matrix(self.Q.toarray()) #needed to remove all the 0's and save memory


    def scan_stability(self, times, disp=True):
        """
        Compute a time scan of the stability
        """
        
        self.set_null_model()
            
        stability_array = []
        number_of_comms_array = []
        community_id_array = []
        MI_array = []
        
        if self.post_process:
            self.Q_matrices = [] #initialise the array to record the exponential
            
        #compute the stability for each time
        for i in tqdm(range(len(times))):
            stability, number_of_comms, community_id, MI_mat, MI = self.run_stability(times[i])

            if self.tpe == 'linearized':
                stability += (1-times[i])

            stability_array.append(stability)
            number_of_comms_array.append(number_of_comms)
            community_id_array.append(community_id)
            MI_array.append(MI)
        
            if disp:
                self.print_single_result(i, len(times))
                
            if self.post_process:
                self.Q_matrices.append(self.Q) 
                
        ttprime = self.compute_ttprime(community_id_array, number_of_comms_array, times)

        #save the results
        timesteps = [element[0] for element in enumerate(times)]
        self.stability_results = pd.DataFrame(
            {
                'Markov time' : times,
                'stability' : stability_array,
                'number_of_communities' : number_of_comms_array,
                'community_id' : community_id_array,
                'MI' : MI_array,
                'ttprime': ttprime
            },
            index = timesteps,
        )
        
        #do the postprocessing here
        if self.post_process:
            print("Apply postprocessing...")
            self.stability_postprocess()
        
        
    def run_stability(self, time):
        """
        run the stability analysis at a given time, only used internally, as it does create the transition matrix
        """

        #compute the adjacency matrix for Louvain
        if self.time_computation:
            start = timer()	

        #self.A_matrix(time)
        self.set_quality_matrix(time)

        if self.time_computation:
            end = timer()	
            print("Matrix exponential:", np.around(end-start,2), "seconds")

        #run Louvain several times and store the communities
        louvain_ensemble = []
        stability_partition_list = []
        number_of_comms_partition_list = []

        if self.time_computation:
            start = timer()

        louvf = partial(louv_f, self.Q, self.null_model)
        with Pool(processes= self.n_processes_louv) as p_stab:  #initialise the parallel computation
            out = p_stab.map(louvf, np.ones(self.louvain_runs)) #run the louvain in parallel

        if self.time_computation:
            end = timer()
            print("Louvain runs:", np.around(end-start,2) , "seconds")

        #re-arange the outputs
        for i in range(self.louvain_runs):
            stability_partition_list.append(out[i][0])
            number_of_comms_partition_list.append(out[i][1])
            louvain_ensemble.append(out[i][2])
            
        self.louvain_ensemble = louvain_ensemble
        
        stability = max(stability_partition_list)
        index = stability_partition_list.index(stability)
        number_of_comms = number_of_comms_partition_list[index]
        community_id = louvain_ensemble[index]
        if self.time_computation:
            start = timer()	

        if self.calcMI:
            MI_mat, MI = self.minfo(louvain_ensemble, stability_partition_list)
            self.MI_mat = MI_mat

        else:
            MI_mat = []
            MI = []
        
        if self.time_computation:
            end = timer()	
            print("MI computations:", np.around(end-start,2), "seconds")

        #save the result
        self.single_stability_result = {
             'stability' : stability,
             'number_of_comms': number_of_comms,
             'community_id': community_id,
             'MI_mat': MI_mat,
             'MI': MI,
             'time': time
            }
        return stability, number_of_comms, community_id, MI_mat, MI

    def run_single_stability(self,time):
        """
        Run one stability analysis at a given time
        """
        
        #self.T_matrix()
        self.set_null_model()
        self.set_quality_matrix(time)
        self.run_stability(time)

# =============================================================================
# MI computations
# =============================================================================
    def minfo(self, louvain_ensemble, stability):
        """
        Compute the mutual information score of several Louvain run
        """
        
        #initialize datasets
        number_of_partitions = len(louvain_ensemble)
        louv_ensemble = np.array(louvain_ensemble) 
        MI_mat = np.zeros((number_of_partitions, number_of_partitions))
        MI = 0 
        n_MI = 0 
        
        if self.all_mi:
            #create the list of louvain pairs for MI
            mi_id= []
            for i in range(number_of_partitions):
                for j in range(i):
                    mi_id.append([i,j])

            mif = partial(mi_f, louv_ensemble)
            with Pool(processes = self.n_processes_mi) as p_mi:  #initialise the parallel computation
                out = p_mi.map(mif, mi_id) #run the MI

            #re-arange the computation
            k=0
            for i in range(number_of_partitions):
                for j in range(i):
                    MI_mat[i,j] = out[k]
                    MI_mat[j,i]= MI_mat[i,j]
                    MI += MI_mat[i,j]
                    n_MI+=1
                    k+=1
        else:
            stability_argsort = np.argsort(stability)
            if len(louvain_ensemble) > self.n_mi:
                louvain_ensemble_reduced = np.array(louvain_ensemble)[stability_argsort[-self.n_mi:]]
            else:
                louvain_ensemble_reduced = louvain_ensemble
 
            MI = 0 
            n_MI = 0 
            for i in range(len(louvain_ensemble_reduced)):
                for j in range(i):
                    MI_mat[i,j] = normalized_mutual_info_score(list(louvain_ensemble[i]),list(louvain_ensemble[j]), average_method='arithmetic' )
                    MI_mat[j,i] = MI_mat[i,j]
                    MI += MI_mat[i,j]
                    n_MI +=1

        return MI_mat, MI/n_MI   
 
# =============================================================================
# postprocessing
# =============================================================================
    def stability_postprocess(self, disp=False):
        """
        Post-process the scan of the stability run
        """

        import os as os
        os.environ["OMP_NUM_THREADS"] = "1"

        #print('Apply the post-processing')
        C = np.vstack(self.stability_results.community_id.values) #store the community label for each time
        N = self.stability_results.number_of_communities.values #store the number of commutities for each time
        S = self.stability_results.stability.values #store the stability of the commutity for each time
        
        times = self.stability_results['Markov time'].values #store the times
        ttprime = self.stability_results['ttprime'].values #store the times

        community_id_array_new = [] #to store the cleaned communities
        stability_array_new = []
        number_of_comms_array_new = []

        #for each time, find the best community structure among all the others
        for i in tqdm(range(len(times))): 
            if disp:
                print('Done ', i ,' of ', len(times))

            Q = self.Q_matrices[i].toarray() #use already computed exponential to save time     

            #compute the Q matrix to sandwich with the community labels
            R = Q 
            for i in range(int(len(self.null_model)/2)):
                R -= np.outer(self.null_model[2*i], self.null_model[2*i+1])

            args = [self.n_neigh, times, i, C, R]

            pprocess_innerf = partial(pprocess_inner_f, args)

            with Pool(processes = self.n_processes_louv) as p_pprocess:  #initialise the parallel computation
                stabilities = p_pprocess.map(pprocess_innerf, range(2*self.n_neigh)) #run the louvain in parallel

            #if linear shift modularity to match the Louvain code
            if self.tpe == 'linearized':
                stabilities += (1-times[i])
                
            #find the best partition for time i, t
            index = np.argmax(stabilities) 
            index_n = index + (i-self.n_neigh)
            
            #record the new partition
            stability_array_new.append(stabilities[index])
            community_id_array_new.append(C[index_n])
            number_of_comms_array_new.append(len(np.unique(C[index_n])))
            
            if disp:
                print('Previous number of comms: ', N[i], ', New number of comms: ', number_of_comms_array_new[-1])
        

        self.stability_results = pd.DataFrame(
            {
                'Markov time' : times,
                'stability' : stability_array_new,
                'number_of_communities' : number_of_comms_array_new,
                'community_id' : community_id_array_new,
                'MI' : self.stability_results.MI.values, 
                'ttprime': ttprime 
            },
            index = self.stability_results.index,
        )
 

    def stability_postprocess_parallel(self, disp=False):
        """
        Post-process the scan of the stability run
        """

        import os as os
        num_threads = os.environ["OMP_NUM_THREADS"]
        os.environ["OMP_NUM_THREADS"] = "1"

        print('Apply the post-processing')
        C = np.vstack(self.stability_results.community_id.values) #store the community label for each time
        N = self.stability_results.number_of_communities.values #store the number of commutities for each time
        S = self.stability_results.stability.values #store the stability of the commutity for each time
        
        times = self.stability_results['Markov time'].values #store the times
        
        #for each time, find the best community structure among all the others
        args = [self.Q_matrices, self.null_model, C, times]
        pprocessf = partial(pprocess_f, args)

        with Pool(processes = self.n_processes_mi) as p_pprocess:  #initialise the parallel computation
            out = p_pprocess.map(pprocessf, range(len(times))) #run the louvain in parallel

        #record the new partition
        community_id_array_new = [] #to store the cleaned communities
        stability_array_new = []
        number_of_comms_array_new = []


        for i in range(len(times)):
            stability_array_new.append(out[i][0])
            community_id_array_new.append(out[i][1])
            number_of_comms_array_new.append(out[i][2])
        
        self.stability_results = pd.DataFrame(
            {
                'Markov time' : times,
                'stability' : stability_array_new,
                'number_of_communities' : number_of_comms_array_new,
                'community_id' : community_id_array_new,
                'MI' : self.stability_results.MI.values
            },
            index = self.stability_results.index,
        )
        os.environ["OMP_NUM_THREADS"] = str(num_threads)


# =============================================================================
# ttprime
# =============================================================================
    def compute_ttprime(self, C, N, T):
        """
        Compute the mutual information score of several Louvain run
        """

        ttprime_id = []
        for t in range(len(T)):
            for tprime in range(len(T)):
                ttprime_id.append([t,tprime])

        ttprimef = partial(ttprime_f, C)
        with Pool(processes= self.n_processes_louv) as p_ttprime:  #initialise the parallel computation
            out = p_ttprime.map(ttprimef, ttprime_id) #run the MI

        ttprime = [] # np.zeros((len(T), len(T)))

        k=0
        for t in range(len(T)):
            ttprime_tmp = np.zeros(len(T))
            for tprime in range(len(T)):
                ttprime_tmp[tprime]= out[k]
                k+=1
            ttprime.append(ttprime_tmp)

        return ttprime
    
             
# =============================================================================
# plotting/printing
# =============================================================================
    def print_single_result(self, i, j ):
        """
        Simple printing function
        """
        print("Step ", i, "/", j) 
        print("T      = ", np.round(self.single_stability_result['time'],3))
        print("N_comm = ", self.single_stability_result['number_of_comms'])
        print("MI     = ", np.round(self.single_stability_result['MI'],3))
        print("Q      = ", np.round(self.single_stability_result['stability'],3))


    def plot_scan(self, time_axis = True):
        """
        Simple plot of a scan
        """

        #get the times paramters
        n_t = len(self.stability_results['ttprime'])
        times = np.log10(self.stability_results['Markov time'].values)

        import matplotlib.gridspec as gridspec
        import matplotlib.ticker as ticker

        plt.figure(figsize=(5,5))

        gs = gridspec.GridSpec(2, 1, height_ratios = [ 1., 0.5])#, width_ratios = [1,0.2] )
        gs.update(hspace=0)
 
        #first plot tt' 
        ax0 = plt.subplot(gs[0, 0])

        #make the ttprime matrix
        ttprime = np.zeros([n_t,n_t])
        for i, tt in enumerate(self.stability_results['ttprime']):
            ttprime[i] = tt 

        ax0.contourf(times, times, ttprime, cmap='YlOrBr')
        ax0.yaxis.tick_left()
        ax0.yaxis.set_label_position('left')
        ax0.set_ylabel(r'$log_{10}(t^\prime)$')
        ax0.axis([times[0],times[-1],times[0],times[-1]])

        #plot the number of clusters
        ax1 = ax0.twinx()
        if time_axis == True:
            ax1.plot(times, self.stability_results['number_of_communities'],c='C0',label='size',lw=2.)
        else:
            ax1.plot(self.stability_results['number_of_communities'],c='C0',label='size',lw=2.)
            
        ax1.yaxis.tick_right()
        ax1.tick_params('y', colors='C0')
        ax1.yaxis.set_label_position('right')
        ax1.set_ylabel('Number of clusters', color='C0')
    
        #make a subplot for stability and MI
        ax2 = plt.subplot(gs[1, 0])

        #first plot the stability
        if time_axis == True:
            ax2.plot(times, self.stability_results['stability'], label=r'$Q$',c='C2')
        else:
            ax2.plot(self.stability_results['stability'], label=r'$Q$',c='C2')

        ax2.set_yscale('log') 
        ax2.tick_params('y', colors='C2')
        ax2.set_ylabel('Modularity', color='C2')
        ax2.yaxis.set_label_position('left')
        #ax2.legend(loc='center right')
        ax2.set_xlabel(r'$log_{10}(t)$')
        
        #ax2.axis([0,n_t,0,self.stability_results.at[0,'number_of_communities']])

        #plot the MMI
        if self.calcMI:
            ax3 = ax2.twinx()
            if time_axis == True:
                ax3.plot(times, self.stability_results['MI'],'-',lw=2.,c='C3',label='MI')
            else:
                ax3.plot(self.stability_results['MI'],'-',lw=2.,c='C3',label='MI')

            ax3.yaxis.tick_right()
            ax3.tick_params('y', colors='C3')
            ax3.set_ylabel(r'Mutual information', color='C3')
            ax3.axhline(1,ls='--',lw=1.,c='C3')
            ax3.axis([times[0], times[-1], 0,1.1])

# =============================================================================
# Sankey diagram
# =============================================================================
    def create_sankey(self, timesteps, comm_sources, comm_allosterics):
        """
        Given a list of times, return a dictonary useable by the sankeywidget
        """
        
        C = self.stability_results['community_id']
        N = self.stability_results['number_of_communities']
        T = self.stability_results['Markov time']

        #first store the nodes for each community at each times
        partitions = {}
        for t in timesteps:

            d = {}
            for x in range(N[t]):
                d["{0}".format(x)] = []

            for i, c in enumerate(C[t]):
                d[str(int(c))].append(str(i))

            partitions["timestep {0}".format(t)] = d
    
        #now create each link between communities, with width proportional to the common number of nodes
        links = []
        for n, t in enumerate(timesteps[:-1]): 

            initial_community =  partitions["timestep {0}". format(timesteps[n])] 
            final_community  =  partitions["timestep {0}". format(timesteps[n+1])]
            initial_comms = N[timesteps[n]]
            final_comms = N[timesteps[n+1]]

            for i in range(int(initial_comms)):
                for j in range(int(final_comms)):
            
                    initial_list = initial_community[str(i)]
                    final_list = final_community[str(j)]

                    value = len(set(initial_list).intersection(final_list))
                    
                    if i in comm_allosterics[n] and j in comm_allosterics[n+1] and i in comm_sources[n] and j in comm_sources[n+1]:
                        color='salmon'
                    elif i in comm_sources[n] and j in comm_sources[n+1]:
                        color='green'
 
                    elif i in comm_allosterics[n] and j in comm_allosterics[n+1]:
                        color='magenta'


                    else:
                        color='steelblue'
                    link = {'source': "T {0}: c {1}".format(n,i), 
                            'target': "T {0}: c {1}".format(n+1,j),
                            'value': value, 
                            'color': color}
                
                    links.append(link)

        return links


# =============================================================================
# function for parallel computations
# =============================================================================
def pprocess_inner_f(args,j):
    n_neigh = args[0]
    times = args[1]
    i = args[2]
    C = args[3]
    R = args[4]

    #compute the stabilities of other partitions 
    j_n = j + (i-n_neigh)
    if j_n<len(times) and j_n>=0: #don't try to compute it for time outside the interval
        #create H matrix
        nr_nodes = len(C[j_n])
        cols = C[j_n].astype(int)
        rows = np.arange(0, nr_nodes , 1)
        data = np.ones(nr_nodes)
        H = sc.sparse.csr_matrix((data, (rows, cols))).toarray()
        
        #compute stability
        stabilities =  np.trace( (H.T).dot(R).dot(H) )
    else:
        stabilities = -1. 

    return stabilities 

def pprocess_f(args, i):
        Q_matrices = args[0]
        null_model = args[1]
        C = args[2]
        times = args[3] 

        Q = Q_matrices[i].toarray() #use already computed exponential to save time
        t = times[i]

        #compute the Q matrix to sandwich with the community labels
        #if tpe == 'linear':
        #    Q =  t*A - np.outer(pi.T, pi) #use time here instead
        #else:                    
        #    Q = A - np.outer(pi.T, pi)
        
        R = Q
        for i in range(int(len(null_model)/2)):
            R =- np.outer(null_model[2*i], null_model[2*i+1])

        stabilities = np.zeros(len(times)) #to record the stability value of the other communities
        #compute the staiblities of other partitions (we could paralellise this loop, but it is fast enough now)
        for j in range(len(times)):
            #create H matrix
            nr_nodes = len(C[j])
            cols = C[j].astype(int)
            rows = np.arange(0, nr_nodes , 1)
            data = np.ones(nr_nodes)
            H = sc.sparse.csr_matrix((data, (rows, cols))).toarray()
            
            #compute stability
            stabilities[j] =  np.trace( (H.T).dot(R).dot(H) )
            
        #if linear shift modularity to match the Louvain code
        #if tpe == 'linear':
        #    stabilities += (1-t)
            
        #find the best partition for time i, t
        index = np.argmax(stabilities)
        return  stabilities[index], C[index], len(np.unique(C[index]))
    
   
def louv_f(Q, null_model, time):
        """
        Function to run in parallel for Louvain evaluations
        """

        non_zero = Q.nonzero()
        from_vec = non_zero[0]
        to_vec = non_zero[1]
        w_vec = Q[non_zero]
        n_edges = len(from_vec)

        null_model_input = array('d', null_model.flatten())
        num_null_vectors = np.shape(null_model)[0] 
        time = 1 #set the time to 1

        stability, community_id = clq.run_louvain(from_vec, to_vec, w_vec, n_edges, null_model_input, num_null_vectors, time)

        community_id = np.array(community_id)  #convert to array

        #stability = np.float(np.loadtxt('data/stability_value_'+str(proc_id)+'.dat'))
        #os.remove('data/stability_value_'+str(proc_id)+'.dat')

        number_of_comms = len(set(community_id))

        return stability, number_of_comms, community_id
        
       
def mi_f(louv_ensemble, args):
        """
        Function to run in paralell for MI evaluations
        """
        
        return normalized_mutual_info_score(louv_ensemble[args[0]],louv_ensemble[args[1]], average_method='arithmetic')

def ttprime_f(C, args):
        """
        Function to run in paralell for ttprime evaluations
        """
        
        return normalized_mutual_info_score(C[args[0]],C[args[1]], average_method='arithmetic' )