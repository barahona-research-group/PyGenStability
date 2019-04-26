import numpy as np

import sys as sys
import numpy as np
import scipy as sc
import pandas as pd

import pylab as plt
import networkx as nx
from sklearn.metrics.cluster import normalized_mutual_info_score
from functools import partial
import pickle 
from multiprocessing import Pool
from multiprocessing import current_process
import os 
from timeit import default_timer as timer
from tqdm import tqdm


import louvainLNL as lv


class Stability(object):
    """ 
    Class to compute various community detection of a networkx Graph
    """

    def __init__(self, G, tpe, louvains_runs, precision, calcMI):
        
        self.G = G # graph
        self.A = nx.adjacency_matrix(self.G, weight='weight')
        self.n = len(G.nodes) # number of nodes 
        self.m = len(G.edges) # number of edges 

        self.tpe = tpe #type of stability, linear or Markov
        self.louvain_runs = louvains_runs #number of Louvain run 
        self.precision = precision #precision threshold for Markov stability

        self.calcMI = calcMI #compute the Mutual info score 
        self.all_mi = False
        self.n_mi = 5

        #self.T = [] #transition matrix Id - D^{-1} A
        #self.pi = [] #stationary solution
        #self.A = [] #adjacency matrix
        #self.A_mat = [] #adjacency matrix to give to Louvain
        
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
           

        if not os.path.isdir('data'):
            os.mkdir('data')
        else:
            import shutil
            shutil.rmtree('data')
            os.mkdir('data')

    def set_null_model(self):
        """
        create the null models
        """

        if self.tpe is 'continuous_combinatorial':
            self.pi = np.ones(self.n)/self.n
            self.null_model = np.array([self.pi, self.pi])

        if self.tpe is 'continuous_normalized' or self.tpe is 'linearized':
            self.degree = np.array(self.A.sum(1)).flatten()
            self.pi = self.degree / self.degree.sum()
            self.null_model = np.array([self.pi, self.pi])

        if self.tpe is 'continuous_signed':
            self.pi = np.zeros(self.n) 
            self.null_model = np.array([self.pi, self.pi])

        if self.tpe is 'modularity_signed':
            A = self.A.toarray()
            A_plus = 0.5*(A + abs(A))
            A_neg = -0.5*(A - abs(A))
            deg_plus = np.array(A_plus.sum(1)).flatten()
            deg_neg = np.array(A_neg.sum(1)).flatten()
            self.deg_norm = deg_plus.sum()+deg_neg.sum()
            self.null_model = np.array([deg_plus, deg_plus/deg_plus.sum(), deg_neg, -deg_neg/deg_neg.sum()])/self.deg_norm



    def set_quality_matrix(self, time):
        """
        create the quality matrix 
        """

        if self.tpe is 'continuous_combinatorial':
            L = 1.*nx.laplacian_matrix(self.G)
            l_min = 1.#abs(sc.sparse.linalg.eigs(L, which='SM',k=2)[0][1])

            ex = sc.sparse.linalg.expm(-time/l_min * L.toarray())
            self.Q = sc.sparse.csc_matrix(np.dot(np.diag(self.pi), ex))

        if self.tpe is 'continuous_normalized':
            L = np.diag(1./self.degree).dot(nx.laplacian_matrix(self.G).toarray())

            l_min = 1.#abs(sc.sparse.linalg.eigs(L, which='SM',k=2)[0][1])
            
            ex = sc.sparse.linalg.expm(-time/l_min * L)
            self.Q = sc.sparse.csc_matrix(np.dot(np.diag(self.pi), ex))

        if self.tpe is 'linearized':
            self.Q = time*self.A/self.degree.sum()
        
        if self.tpe is 'continuous_signed':
            A = nx.adjacency_matrix(self.G).toarray()
            degree = abs(A).sum(1)
            D = np.diag(degree)
            L = sc.sparse.csr_matrix(D - A)

            l_min = 1.#abs(sc.sparse.linalg.eigs(L, which='SM',k=2)[0][1])

            ex = sc.sparse.linalg.expm(-time/l_min * L.toarray())
            self.Q = sc.sparse.csc_matrix(ex)


        if self.tpe is 'modularity_signed':
            self.Q = 2*time*self.A/self.deg_norm


        self.Q = (np.max(self.Q)*self.precision)*np.round(self.Q/((np.max(self.Q)*self.precision)))
        self.Q = sc.sparse.csc_matrix(self.Q.toarray()) #needed to remove all the 0's and save memory

    def save_quality_null(self):
        """
        save files for generalised Louvain cpp code
        """


        nx.write_weighted_edgelist(nx.Graph(self.Q), 'quality.edj' )

        np.savetxt('null_model.edj', self.null_model.T)



    def print_single_result(self, i, j ):
        """
        Simple printing function
        """
        print("Step ", i, "/", j) 
        print("T      = ", np.round(self.single_stability_result['time'],3))
        print("N_comm = ", self.single_stability_result['number_of_comms'])
        print("MI     = ", np.round(self.single_stability_result['MI'],3))
        print("Q      = ", np.round(self.single_stability_result['stability'],3))


    def scan_stability(self, times, disp=True):
        """
        Compute a time scan of the stability
        """
        
        #compute transition matrix and stationary distribution
        #self.T_matrix()
        #self.stationary_distribution()

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

            if self.tpe is 'linearized':
                stability += (1-times[i])

            stability_array.append(stability)
            number_of_comms_array.append(number_of_comms)
            community_id_array.append(community_id)
            MI_array.append(MI)
        
            if disp:
                self.print_single_result(i, len(times))
                
            if self.post_process:
                self.Q_matrices.append(self.Q) 
                

        #save the results
        timesteps = [element[0] for element in enumerate(times)]
        self.stability_results = pd.DataFrame(
            {
                'Markov time' : times,
                'stability' : stability_array,
                'number_of_communities' : number_of_comms_array,
                'community_id' : community_id_array,
                'MI' : MI_array
            },
            index = timesteps,
        )
        
        self.ttprime = self.compute_ttprime()

        #do the postprocessing here
        if self.post_process:
            self.stability_postprocess()
        
    def compute_ttprime(self):
        """
        Compute the mutual information score of several Louvain run
        """
        C = self.stability_results['community_id']
        N = self.stability_results['number_of_communities']
        T = self.stability_results['Markov time']
        Tprime = self.stability_results['Markov time']

        ttprime_id = []
        for t in range(len(T)):
            for tprime in range(len(T)):
                ttprime_id.append([t,tprime])

        ttprimef = partial(ttprime_f, C)
        with Pool(processes= self.n_processes_louv) as p_ttprime:  #initialise the parallel computation
            out = p_ttprime.map(ttprimef, ttprime_id) #run the MI

        ttprime = np.zeros((len(T), len(T)))

        k=0
        for t in range(len(T)):
            for tprime in range(len(T)):
                ttprime[t,tprime]= out[k]
                k+=1

        return ttprime
    
             
    def plot_scan(self, time_axis = True):
        """
        Simple plot of a scan
        """

        #self.compute_ttprime()
        fig, ax0 = plt.subplots()
        
        t_max = len(self.ttprime)

        ax0.imshow(self.ttprime,aspect='auto', cmap='YlOrBr',vmin=0.0,vmax=1.1,alpha=0.6,extent=[0,t_max,0,t_max])#,origin='bottom')
        ax0.get_yaxis().set_visible(False)
        ax0.axis([0,len(self.ttprime),0,t_max])
        ax1 = ax0.twinx()

        if time_axis == True:
            ax1.semilogx(self.stability_results['Markov time'], self.stability_results['stability'], label=r'$Q$',c='C2')
            if self.calcMI:
                ax1.semilogx(self.stability_results['Markov time'], self.stability_results['MI'],'-+',lw=2.,c='C3',label='MI')
        else:
            ax1.plot(self.stability_results['stability'], label=r'$Q$',c='C2')

            if self.calcMI:
                ax1.plot(self.stability_results['MI'],'-',lw=2.,c='C3',label='MI')
            
        ax1.axhline(1,ls='--',lw=1.,c='C3')

        ax1.yaxis.tick_right()
        ax1.set_ylabel(r'$MI,Q$')
        ax1.yaxis.set_label_position('right')
        ax1.legend(loc='center right')

        ax1.axis([0, t_max, 0, 1.05])

        ax2 = ax0.twinx()
        
        if time_axis == True:
            ax2.plot(self.stability_results['Markov time'], self.stability_results['number_of_communities'],c='C0',label='size',lw=2.)
        else:
            ax2.plot(self.stability_results['number_of_communities'],c='C0',label='size',lw=2.)
            
        ax2.yaxis.tick_left()
        ax2.yaxis.set_label_position('left')
        ax2.set_ylabel(r'$size$')
        ax2.set_xlabel(r'$time$')
    
        ax2.legend(loc='center left')
        ax2.axis([0,t_max,0,self.stability_results.at[0,'number_of_communities']])

    def edge_list_from_A(self):
        """
        Convert an adjacency matrix to an array readable by the Louvain code
        """
        
        #X and Y are swaped to fit the C code
        Y, X, W = sc.sparse.find(self.A_mat)
        
        return np.array(np.stack((X,Y,W)),dtype=np.float64)

   
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
    
    def run_stability(self, time):
        """
        run the stability analysis at a given time, only used internally, as it does create the transition matrix
        """

        #compute the adjacency matrix for Louvain
        if self.time_computation:
            start = timer()	

        #self.A_matrix(time)
        self.set_quality_matrix(time)
        self.save_quality_null()

        if self.time_computation:
            end = timer()	
            print("Matrix exponential:", np.around(end-start,2), "seconds")


        #run Louvain several times and store the communities
        louvain_ensemble = []
        stability_partition_list = []
        number_of_comms_partition_list = []

        #if self.tpe is 'linear': #in this case, use the time directly in the Louvain code
        #    #args = [self.edge_list_from_A(), self.precision] #arguments for louvain
        #    args = [self.G, self.precision] #arguments for louvain
        #    t = 1#time
        #else:
        #    #args = [self.edge_list_from_A(), self.precision] #arguments for louvain
        #    args = [self.G, self.precision] #arguments for louvain
        #    t = 1


        if self.time_computation:
            start = timer()

        louvf = partial(louv_f, self.G)
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
        #self.stationary_distribution()
        self.set_null_model()
        self.set_quality_matrix(time)
        self.save_quality_null()
        self.run_stability(time)
        

    def run_single_stability_singlecore(self,time):
        """
        Run one stability analysis at a given time
        Old function, not using parallel computations
        """
        
        self.T_matrix()
        self.stationary_distribution()
        self.run_stability_singlecore(time)
        
        
    def run_stability_singlecore(self, time):
        """
        run the stability analysis at a given time, only used internally, as it does create the transition matrix
        Old function, not using parallel computations

        """
        
        #compute the adjacency matrix for Louvain
        self.A_matrix(time)

        #run Louvain several times and store the communities
        louvain_ensemble = []
        stability_partition_list = []
        number_of_comms_partition_list = []

        for i in range(self.louvain_runs):
            
            if self.tpe is 'linear': #in this case, use the time directly in the Louvain code
                (stability, number_of_comms, community_id) = lv.stability(self.edge_list_from_A(), time, self.precision)
            else:
                (stability, number_of_comms, community_id) = lv.stability(self.edge_list_from_A(), 1. , self.precision)
    
            louvain_ensemble.append(community_id)
            stability_partition_list.append(stability)
            number_of_comms_partition_list.append(number_of_comms)
        

        stability = max(stability_partition_list)
        index = stability_partition_list.index(stability)
        number_of_comms = number_of_comms_partition_list[index]
        community_id = louvain_ensemble[index]

        
        #compute the MI
        if self.calcMI:
            MI_mat, MI = self.minfo_singlecore(louvain_ensemble,stability)

        else:
            MI_mat = []
            MI = []
        
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
         
    def minfo_singlecore(self, louvain_ensemble):
        """
        Compute the mutual information score of several Louvain run
        Old function, not using parallel computations

        """
        
        number_of_partitions = len(louvain_ensemble)
        louvain_ensemble = np.array(louvain_ensemble) #convert to numpy array
        
        MI_mat = np.zeros((number_of_partitions, number_of_partitions))
    
        MI = 0 
        n_MI = 0 
        for i in range(number_of_partitions):
            for j in range(i):
                MI_mat[i,j] = normalized_mutual_info_score(list(louvain_ensemble[i]),list(louvain_ensemble[j]), average_method='arithmetic' )
                MI_mat[j,i]= MI_mat[i,j]
                MI += MI_mat[i,j]
                n_MI+=1

        return MI_mat, MI/n_MI
 
    def stability_postprocess(self, disp=False):
        """
        Post-process the scan of the stability run
        """

        import os as os
        os.environ["OMP_NUM_THREADS"] = "1"

        print('Apply the post-processing')
        C = np.vstack(self.stability_results.community_id.values) #store the community label for each time
        N = self.stability_results.number_of_communities.values #store the number of commutities for each time
        S = self.stability_results.stability.values #store the stability of the commutity for each time
        
        times = self.stability_results['Markov time'].values #store the times
        
        community_id_array_new = [] #to store the cleaned communities
        stability_array_new = []
        number_of_comms_array_new = []

        #for each time, find the best community structure among all the others
        for i in tqdm(range(len(times))): 
            if disp:
                print('Done ', i ,' of ', len(times))

            Q = self.Q_matrices[i].toarray() #use already computed exponential to save time
            

            #compute the Q matrix to sandwich with the community labels
            #if self.tpe is 'linear':
            #    R =  Q - np.outer(self.pi_1, self.pi_2) #use time here instead
            #else:                    
            R = Q 
            for i in range(int(len(self.null_model)/2)):
                R -= np.outer(self.null_model[2*i], self.null_model[2*i+1])

            args = [self.n_neigh, times, i, C, R]

            pprocess_innerf = partial(pprocess_inner_f, args)

            with Pool(processes = self.n_processes_louv) as p_pprocess:  #initialise the parallel computation
                stabilities = p_pprocess.map(pprocess_innerf, range(2*self.n_neigh)) #run the louvain in parallel

            #if linear shift modularity to match the Louvain code
            if self.tpe is 'linearized':
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
                'MI' : self.stability_results.MI.values
            },
            index = self.stability_results.index,
        )
 
              

    def stability_postprocess_single(self, disp=False):
        """
        Post-process the scan of the stability run
        """
        import os as os
        os.environ["OMP_NUM_THREADS"] = "1"

        print('Apply the post-processing')
        C = np.vstack(self.stability_results.community_id.values) #store the community label for each time
        N = self.stability_results.number_of_communities.values #store the number of commutities for each time
        S = self.stability_results.stability.values #store the stability of the commutity for each time
        
        times = self.stability_results['Markov time'].values #store the times
        
        community_id_array_new = [] #to store the cleaned communities
        stability_array_new = []
        number_of_comms_array_new = []

        #for each time, find the best community structure among all the others
        for i, t in enumerate(times): 
            if disp:
                print('Done ',i ,' of ', len(times))

            A = self.A_matrices[i].toarray() #use already computed exponential to save time
            

            #compute the Q matrix to sandwich with the community labels
            if self.tpe is 'linear':
                Q =  t*A - np.outer(self.pi.T, self.pi) #use time here instead
            else:                    
                Q = A - np.outer(self.pi.T, self.pi)

    
            """
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
                stabilities[j] =  np.trace( (H.T).dot(Q).dot(H) )
            """  

            #if linear shift modularity to match the Louvain code
            if self.tpe is 'linearized':
                stabilities += (1-t)
                
            #find the best partition for time i, t
            index = np.argmax(stabilities)
            
            #record the new partition
            stability_array_new.append(stabilities[index])
            community_id_array_new.append(C[index])
            number_of_comms_array_new.append(len(np.unique(C[index])))
            
            if disp:
                print('Previous number of comms: ', N[i], ', New number of comms: ', number_of_comms_array_new[-1])
        

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













    def T_matrix_old(self):
        """
        compute the transition matrix
        """
    
        #adjacency matrix
        self.A = nx.adjacency_matrix(self.G, weight='weight')
    
        #weighted degree
        d = np.asarray(sc.sparse.csr_matrix.sum(self.A, 1).flatten())[0]

        if self.tpe is 'markov_comb': #save the diagonal matrix for the combinatorial Laplacian
            self.D = sc.sparse.spdiags(d,[0],len(d),len(d), format='csr')
    
        #if d==0, set its inverse to 1
        d[d==0]=1
        Dinv = sc.sparse.spdiags(1./d,[0],len(d), len(d), format='csr')
   
        if self.rw_tpe == 'max':
            
            w, v = sc.sparse.linalg.eigs(1.*self.A, 1)
            lamb = np.max(w)
            psi = v[:,np.argmax(w)]
            
            self.T = sc.sparse.csr_matrix(np.diag(psi).dot(self.A.toarray()).dot(np.diag(1./psi))/lamb)

        else:
            self.T = np.dot(Dinv, self.A)
        
    def stationary_distribution_old(self, k=2):
        """
        Compute the stationary distribution
        """
        
        if self.tpe == 'markov_comb':
            self.pi = np.ones(len(self.G))/len(self.G)

        if self.tpe == 'markov_norm' or self.tpe == 'linear':
            w,vl = sc.sparse.linalg.eigs(self.T.T,k)
            v= np.real(vl.transpose()[np.argmax(np.real(w))])
        
            if np.max(np.real(w)) < 1.-1e-10:
                print("Erreur!! Eigenvalues:",w,"Choice:",np.argmax(np.real(w)))


            self.pi = v/v.sum()

           
    def A_matrix_old(self, time):
        """
        Create symetric adjacency matrix to give to Louvain
        """
        
        n, m = self.T.shape
        Id = sc.sparse.eye(m, n, format='csr')
        Pi = sc.sparse.spdiags(self.pi, [0], m, n, format='csr')

        if self.tpe is 'markov_norm':
            ex = sc.sparse.linalg.expm(time*sc.sparse.csc_matrix(self.T-Id))
            A_tmp = np.dot(Pi,ex)
            
        elif self.tpe is 'markov_comb':
            L = self.D - self.A 
            l_min = abs(sc.sparse.linalg.eigs(L, which='SM',k=2)[0][1])

            ex = sc.sparse.csc_matrix( sc.sparse.linalg.expm(-time*L.toarray()))
            A_tmp = np.dot(Pi,ex)
        
        elif self.tpe is 'linear':
            A_tmp = np.dot(Pi,self.T)
            
        
        self.A_mat = 0.5*(A_tmp + A_tmp.T)

        #remove small-valued edges
        self.A_mat = (np.max(self.A_mat)*self.precision)*np.round(self.A_mat/((np.max(self.A_mat)*self.precision)))
        self.A_mat = sc.sparse.csc_matrix(self.A_mat.toarray()) #needed to remove all the 0's and save memory







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



   
def run_gen_louvain(G, time, proc_id):

    import os as os
    os.system('/home/arnaudon/codes/MarkovStability/gen_cpp/run_gen_louvain.sh quality.edj null_model.edj ' + str(time) + ' ' + str(np.random.randint(1000)) + ' ' + str(proc_id))
    partitions = np.loadtxt('data/optimal_partitions_'+str(proc_id)+'.dat')
    os.remove('data/optimal_partitions_'+str(proc_id)+'.dat')
    if len(np.shape(partitions))>1:
        partitions = partitions[-1]
        
    return partitions 

def louv_f(G, time):
        """
        Function to run in parallel for Louvain evaluations
        """

        current = current_process()
        proc_id = np.int(current._identity[0])
        #(stability, number_of_comms, community_id) = lv.stability(args[0],time, args[1])

        community_id = run_gen_louvain(G, time, proc_id)
        stability = np.float(np.loadtxt('data/stability_value_'+str(proc_id)+'.dat'))
        os.remove('data/stability_value_'+str(proc_id)+'.dat')
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
    

