# *PyGenStability*

This ``python`` package is designed for multiscale community detection with Markov Stability (MS) analysis [1, 2, 3, 4] and allows researchers to identify robust network partitions at different resolutions. It implements several variants of the MS cost functions that are based on graph diffusion processes to explore the network (see illustration below). Whilst primarily built for MS, the internal architecture of *PyGenStability* has been designed to solve for a wide range of clustering cost functions since it is based on optimising the so-called generalized Markov Stability function [5]. To maximize the generalized Markov Stability cost function, *PyGenStability* provides a convenient ``python`` interface for ``C++`` implementations of Louvain [6] and Leiden [7] algorithms.
We further provide specific analysis tools to process and analyse the results from multiscale community detection, and to facilitate the automatic selection of robust partitions [8]. *PyGenStability* is accompanied by a software paper that further details the implementation, result analysis, benchmarks and applications [9].

![illustration](../artwork/diffusion_schematic.png)

## Installation

You can install the package using [pypi](https://pypi.org/project/PyGenStability/):

```bash
pip install pygenstability
```

Using a fresh python3 virtual environment, e.g. conda, may be recommended to avoid conflicts with other python packages. 

By default, the package uses the Louvain algorithm [6] for optimizing generalized Markov Stability. To use the Leiden algorithm [7], install this package with:
```bash
pip install pygenstability[leiden]
```

To plot network partitions using `networkx`, install this package with:
```bash
pip install pygenstability[networkx]
```

To use `plotly` for interactive plots in the browser, install this package with: 
```bash
pip install pygenstability[plotly]
```

To install all dependencies, run:
```bash
pip install pygenstability[all]
```

### Installation from GitHub

You can also install the source code of this package from GitHub directly by first cloning this repo with:
```bash
git clone --recurse-submodules https://github.com/ImperialCollegeLondon/PyGenStability.git
```

(if the `--recurse-submodules` has not been used, just do `git submodule update --init --recursive` to fetch the submodule with M. Schaub's code).

The wrapper for the submodule uses Pybind11 https://github.com/pybind/pybind11 and, to install the package, simply run (within the `PyGenStability` directory):
```bash
pip install . 
```
using a fresh python3 virtual environment to avoid conflicts. Similar to above, you can also specify additional dependencies, e.g. to install the package with `networkx` run:
```bash
pip install .[networkx]
```

## Using the code

The code is simple to run with the default settings. We can input our graph (of type scipy.csgraph), run a scan in scales with a chosen Markov Stability constructor and plot the results in a summary figure presenting different partition quality measures across scales (values of MS cost function, number of communities, etc.) with an indication of optimal scales.

```python
import pygenstability as pgs
results = pgs.run(graph)
pgs.plot_scan(results)
```

Although it is enforced in the code, it is advised to set environment variables
```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
```
to ensure numpy does not use multi-threadings, which may clash with the parallelisation and slow down the computation. 

There are a variety of further choices that users can make that will impact the partitioning, including:
- Constructor: Generalized Markov Stability requires the user to input a quality matrix and associated null models. We provide an object-oriented module to write user-defined constructors for these objects, with several already implemented (see `pygenstability/constructors.py` for some classic examples).
- Generalized Markov Stability maximizers: To maximize the NP-hard optimal generalized Markov Stability we interface with two algorithms: (i) Louvain and (ii) Leiden.

While Louvain is defined as the default due to its familiarity within the research community, Leiden is known to produce better partitions and can be used by specifying the run function. Note that Leiden may perform significantly faster for **large** graphs.

```python
results = pgs.run(graph, method="leiden")
```

There are also additional post-processing and analysis functions, including:
- Plotting via matplotlib and plotly (interactive).
- Automated optimal scale selection.

Optimal scale selection [8] is performed by default with the run function but can be repeated with different parameters if needed, see `pygenstability/optimal_scales.py`. To reduce noise, e.g., one can increase the parameter values for `block_size` and `window_size`. The optimal network partitions can then be plotted given a NetworkX nx_graph.

```python
results = pgs.identify_optimal_scales(results, kernel_size=10, window_size=5)
pgs.plot_optimal_partitions(nx_graph, results)
```

## Constructors
 
We provide an object-oriented module for constructing quality matrices and null models in `pygenstability/constructors.py`. Various constructors are implemented for different types of graphs:

- `linearized` based on linearized MS for large undirected weighted graphs [3]
- `continuous_combinatorial` based on combinatorial Laplacian for undirected weighted graphs [3]
- `continuous_normalized` based on random-walk normalized Laplacians for undirected weighted graphs [3]
- `signed_modularity` based on signed modularity for large signed graphs [10]
- `signed_combinatorial` based on signed combinatorial Laplacian for signed graphs [5]
- `directed` based on random-walk Laplacian with teleportation for directed weighted graphs [3]
- `linearized_directed` based on random-walk Laplacian with teleportation for large directed weighted graphs

For the computationally efficient analysis of **large** graphs, we recommend using the `linearized`, `linearized_directed` or `signed_modularity` constructors instead of `continuous_combinatorial`, `continuous_normalized`, `directed` or `signed_combinatorial` that rely on the computation of matrix exponentials.

For those of you that wish to implement their own constructor, you will need to design a function with the following properties:

- take a scipy.csgraph `graph` and a float `time` as argument
- return a `quality_matrix` (sparse scipy matrix) and a `null_model` (multiples of two, in a numpy array)

## Graph-based data clustering

PyGenStability can also be used to perform multiscale graph-based data clustering on data that comes in the form of a sample-by-feature matrix. This approach was shown to achieve better performance than other popular clustering methods without the need of setting the number of clusters externally [11]. 

We provide an easy-to-use interface in our `pygenstability.data_clustering.py` module. Given a sample-by-feature matrix `X`, one can apply graph-based data clustering as follows:

```python
clustering = pgs.DataClustering(
    graph_method="cknn-mst",
    k=5,
    constructor="continuous_normalized")

# apply graph-based data clustering to X
results = clustering.fit(X)

# identify optimal scales and plot scan
clustering.scale_selection(kernel_size=0.2)
clustering.plot_scan()
```

We currently support $k$-Nearest Neighbor (kNN) and Continuous $k$-Nearest Neighbor (CkNN) [12] graph constructions (specified by `graph_method`) augmented with the minimum spanning tree to guarentee connectivity, where `k` refers to the number of neighbours considered in the construction. See documentation for a list of all parameters. All functionalities of PyGenStability including plotting and scale selection are also available for data clustering. For example, given two-dimensional coordinates of the data points one can plot the optimal partitions directly:

```python
# plot robust partitions
clustering.plot_robust_partitions(x_coord=x_coord,y_coord=y_coord)
```

## Contributors

- Alexis Arnaudon, GitHub: `arnaudon <https://github.com/arnaudon>`
- Robert Peach, GitHub: `peach-lucien <https://github.com/peach-lucien>`
- Juni Schindler, GitHub: `juni-schindler <https://github.com/juni-schindler>`

We always look out for individuals that are interested in contributing to this open-source project. Even if you are just using *PyGenStability* and made some minor updates, we would be interested in your input. 

## Cite

Please cite our paper if you use this code in your own work:

```
@article{pygenstability,
  author = {Arnaudon, Alexis and Schindler, Juni and Peach, Robert L. and Gosztolai, Adam and Hodges, Maxwell and Schaub, Michael T. and Barahona, Mauricio},  
  title = {Algorithm 1044: PyGenStability, a Multiscale Community Detection Framework with Generalized Markov Stability},
  journal = {ACM Trans. Math. Softw.},
  volume = {50},
  number = {2},
  pages = {15:1–15:8}
  year = {2024},
  doi = {10.1145/3651225}
}
```

The original paper for Markov Stability can also be cited as:

```
@article{delvenne2010stability,
  title={Stability of graph communities across time scales},
  author={Delvenne, J-C and Yaliraki, Sophia N and Barahona, Mauricio},
  journal={Proceedings of the National Academy of Sciences},
  volume={107},
  number={29},
  pages={12755--12760},
  year={2010},
  publisher={National Acad Sciences}
}
```


## Run example


In the `example` folder, a demo script with a stochastic block model can be tried with 

```bash
python simple_example.py
```
or using the click app:
```bash
./run_simple_example.sh
```



Other examples can be found as jupyter-notebooks in the `examples/` directory, including:

- Example 1: Undirected SBM
- Example 2: Multiscale SBM
- Example 3: Directed networks
- Example 4: Custom constructors
- Example 5: Hypergraphs
- Example 6: Signed networks
- Example 7: Graph-based data clustering

Finally, we provide applications to real-world networks in the `examples/real_examples/` directory, including:

- Power grid network
- Protein structures


## Our other available packages

If you are interested in trying our other packages, see the below list:

- [GDR](https://github.com/barahona-research-group/GDR) : Graph diffusion reclassification. A methodology for node classification using graph semi-supervised learning.
- [hcga](https://github.com/barahona-research-group/hcga) : Highly comparative graph analysis. A graph analysis toolbox that performs massive feature extraction from a set of graphs, and applies supervised classification methods.
- [MSC](https://github.com/barahona-research-group/MultiscaleCentrality) : MultiScale Centrality: A scale-dependent metric of node centrality.
- [DynGDim](https://github.com/barahona-research-group/DynGDim) : Dynamic Graph Dimension: Computing the relative, local and global dimension of complex networks.
- [RMST](https://github.com/barahona-research-group/RMST) : Relaxed Minimum Spanning Tree: Computing the relaxed minimum spanning tree to sparsify networks whilst retaining dynamic structure.
- [StEP](https://github.com/barahona-research-group/StEP) : Spatial-temporal Epidemiological Proximity: Characterising contact in disease outbreaks via a network model of spatial-temporal proximity.

## References

[1] J.-C. Delvenne, S. N. Yaliraki, and M. Barahona, 'Stability of graph communities across time scales', *Proceedings of the National Academy of Sciences*, vol. 107, no. 29, pp. 12755–12760, Jul. 2010, doi: 10.1073/pnas.0903215107. Originally arXiv:0812.1811 (2008)

[2] R. Lambiotte, J.-C. Delvenne, and M. Barahona, 'Laplacian dynamics and multiscale modular structure in networks', arXiv preprint arXiv:0812.1770 (9 Dec 2008)

[3] R. Lambiotte, J.-C. Delvenne, and M. Barahona, 'Random Walks, Markov Processes and the Multiscale Modular Organization of Complex Networks', *IEEE Trans. Netw. Sci. Eng.*, vol. 1, no. 2, pp. 76–90, Jul. 2014, doi: 10.1109/TNSE.2015.2391998.

[4] J.-C. Delvenne, M. T. Schaub, S. N. Yaliraki, and M. Barahona, 'The stability of a graph partition: A dynamics-based framework for community detection', *Dynamics On and Of Complex Networks*, vol. 2, pp. 221-242, Springer New York, Apr 2013 - arxiv.org/abs/1308.1605

[5] M. T. Schaub, J.-C. Delvenne, R. Lambiotte, and M. Barahona, 'Multiscale dynamical embeddings of complex networks', *Phys. Rev. E*, vol. 99, no. 6, Jun. 2019, doi: 10.1103/PhysRevE.99.062308.

[6] V. D. Blondel, J.-L. Guillaume, R. Lambiotte, and E. Lefebvre, 'Fast unfolding of communities in large networks', *J. Stat. Mech.*, vol. 2008, no. 10, Oct. 2008, doi: 10.1088/1742-5468/2008/10/p10008.

[7] V. A. Traag, L. Waltman, and N. J. van Eck, 'From Louvain to Leiden: guaranteeing well-connected communities', *Sci Rep*, vol. 9, no. 1, p. 5233, Mar. 2019, doi: 10.1038/s41598-019-41695-z.

[8] J. Schindler, J. Clarke, and M. Barahona, 'Multiscale Mobility Patterns and the Restriction of Human Movement', *Royal Society Open Science*, vol. 10, no. 10, p. 230405, Oct. 2023, doi: 10.1098/rsos.230405.

[9] A. Arnaudon, J. Schindler, R. L. Peach, A. Gosztolai, M. Hodges, M. T. Schaub, and M. Barahona, 'Algorithm 1044: PyGenStability, a Multiscale Community Detection Framework with Generalized Markov Stability', *ACM Trans. Math. Softw.*, vol. 50, no. 2, p. 15:1–15:8, Jun. 2024, doi: 10.1145/3651225.

[10] S. Gomez, P. Jensen, and A. Arenas, 'Analysis of community structure in networks of correlated data'. *Physical Review E*, vol. 80, no. 1, p. 016114, Jul. 2009, doi: 10.1103/PhysRevE.80.016114.

[11] Z. Liu and M. Barahona, 'Graph-based data clustering via multiscale community detection', *Applied Network Science*, vol. 5, no. 1, p. 3, Dec. 2020, doi: 10.1007/s41109-019-0248-7.

[12] T. Berry and T. Sauer, 'Consistent manifold representation for topological data analysis', *Foundations of Data Science*, vol. 1, no. 1, p. 1-38, Feb. 2019, doi: 10.3934/fods.2019001.

## Licence

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see http://www.gnu.org/licenses/.


