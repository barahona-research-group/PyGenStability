# *PyGenStability*
This ``python`` package is designed for multiscale community detection with Markov Stability (MS) analysis [1, 2] and allows researchers to identify robust network partitions at different resolutions. It implements several variants of the MS cost functions that are based on graph diffusion processes. Whilst primarily built for MS, the internal architecture of has been designed to solve for a wide range of clustering cost functions since it is based on optimising the so-called generalised modularity function [3]. To maximize the generalized modularity cost function, *PyGenStability* provides a convenient ``python`` interface for ``C++`` implementations of Louvain [4] and Leiden [5] algorithms.
We further provide specific analysis tools to process and analyse the results from multiscale community detection, and to facilitate the autmatic detection of robust partitions [6]. *PyGenStability* is accompanied by a software paper that further details the implementation, result analysis, benchmarks and applications [7].

<p align="center">
  <img src="docs/artwork/diffusion_schematic.png" width="800" />
</p>


## Installation


The wrapper uses Pybind11 https://github.com/pybind/pybind11 and the package can simply be installed by first cloning this repo with

```
git clone --recurse-submodules https://github.com/ImperialCollegeLondon/PyGenStability.git
```

(if the `--recurse-submodules` has not been used, just do `git submodule update --init --recursive` to fetch the submodule with M. Schaub's code). 

Then, to install the package, simply run
```
pip install . 
```
using a fresh `virtualenv` in python3 may be recommanded to avoid conflict of python packages. 

To use plotly for interacting plos in browser, install this package with 
```
pip install .[plotly]
```

To use contrib module, with additional tools, run
```
pip install .[contrib]
```

To install all dependencies, run
```
pip install .[all]
```

## Using the code

The code is simple to run with the default settings, we can simply import the run and plotting functions, input our graph (of type scipy.csgraph), and then plot the results.

```
from pygenstability import run, plotting
results = run(graph)
plotting.plot_scan(results)
```

There are a variety of further choices that user can make that will impact the partitioning, including:
- Constructor: Generalized modularity requires the user to input a quality matrix and associated null models. We provide an object oriented module to write user-defined constructors for these objects, with several already implemented (see `pygenstability/constructors.py` for some classic examples).
- Generalized modularity maximizers: To maximize the NP-hard optimal generalized modularity we interface with two algorithms: (i) Louvain and (ii) Leiden.

There are also additional postprocessing and analysis functions, including:
- Plotting via matplotlib and plotly (interactive).
- Automated optimal scale detection.

## Custom constructors
 
For those of you that wish to implement their own constructor, you will need to design a function with the following properties:

- take a networkx `graph` and a float `time` as argument
- return a `quality_matrix` (sparse scipy matrix) and a `null_model` (multiples of two, in a numpy array)

Please see `pygenstability/constructors.py` for the existing implemented constructors. 

## Documentation

Documentation is here: https://barahona-research-group.github.io/PyGenStability/

## Contributers

- Alexis Arnaudon, GitHub: `arnaudon <https://github.com/arnaudon>`
- Robert Peach, GitHub: `peach-lucien <https://github.com/peach-lucien>`
- Dominik Schindler, GitHub: `d-schindler <https://github.com/d-schindler>`

We are always on the look out for individuals that are interested in contributing to this open-source project. Even if you are just using *PyGenStability* and made some minor updates, we would be interested in your input. 

## Cite

Please cite our paper if you use this code in your own work:

```
preprint incoming...
```

The original paper for Markov Stability can also be cited as:

```
@article{delvenne2010stability,
  title={Stability of graph communities across time scales},
  author={Delvenne, J-C and Yaliraki, Sophia N and Barahona, Mauricio},
  journal={Proceedings of the national academy of sciences},
  volume={107},
  number={29},
  pages={12755--12760},
  year={2010},
  publisher={National Acad Sciences}
}
```


## Run example


In the `example` folder, a demo script with stochastic block model can be tried with 

```
python simple_example.py
```
 or using the click app:
 ```
 ./run_simple_example.sh
 ```



Other examples can be found as jupyter-notebooks in `examples/` directory, including:
* Example 1: Undirected SBM
* Example 2: Directed networks
* Example 3: Custom constructors
* Example 4: Hypergraphs


## Our other available packages

If you are interested in trying our other packages, see the below list:
* [GDR](https://github.com/barahona-research-group/GDR) : Graph diffusion reclassification. A methodology for node classification using graph semi-supervised learning.
* [hcga](https://github.com/barahona-research-group/hcga) : Highly comparative graph analysis. A graph analysis toolbox that performs massive feature extraction from a set of graphs, and applies supervised classification methods.
* [MSC](https://github.com/barahona-research-group/MultiscaleCentrality) : MultiScale Centrality: A scale dependent metric of node centrality.
* [DynGDim](https://github.com/barahona-research-group/DynGDim) : Dynamic Graph Dimension: Computing the relative, local and global dimension of complex networks.
* [RMST](https://github.com/barahona-research-group/RMST) : Relaxed Minimum Spanning Tree: Computing the relaxed minimum spanning tree to sparsify networks whilst retaining dynamic structure.
* [StEP](https://github.com/barahona-research-group/StEP) :  Spatial-temporal Epidemiological Proximity: Characterising contact in disease outbreaks via a network model of spatial-temporal proximity.

## References



## Licence

