# PyGenStability
Python wrapper of the generalised Louvain code of Michael Schaub at https://github.com/michaelschaub/generalizedLouvain with python code to run various versions of Markov Stability. 

Installation
-------------

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

Documentation
-------------

Documentation is here: https://barahona-research-group.github.io/PyGenStability/

Example
-------

In the `example` folder, a demo script with stochastic block model can be tried with 

```
python simple_example.py
```
 or using the click app:
 ```
 ./run_simple_example.sh
 ```
 
 Custom constructors
 -------------------
 
 The generalized Louvain code needs a constructor, which can be a function with the following properties:
 
 - take a networkx `graph` and a float `time` as argument
 - return a `quality_matrix` (sparse scipy matrix) and a `null_model` (multiples of two, in a numpy array)
 
 Please see `pygenstability/constructors.py` for some classic examples. 


Contrib
-------

This module contains various additional tools one can use. Currently it contains:
 - optimal-scales: to find and plot optimal scales accros time
 - sankey: plot sankey diagrams of clusters accros time



## Our other available packages

If you are interested in trying our other packages, see the below list:
* [GDR](https://github.com/barahona-research-group/GDR) : Graph diffusion reclassification. A methodology for node classification using graph semi-supervised learning.
* [hcga](https://github.com/barahona-research-group/hcga) : Highly comparative graph analysis. A graph analysis toolbox that performs massive feature extraction from a set of graphs, and applies supervised classification methods.
* [MSC](https://github.com/barahona-research-group/MultiscaleCentrality) : MultiScale Centrality: A scale dependent metric of node centrality.
* [DynGDim](https://github.com/barahona-research-group/DynGDim) : Dynamic Graph Dimension: Computing the relative, local and global dimension of complex networks.
* [PyGenStability](https://github.com/barahona-research-group/PyGenStability) : Markov Stability: Computing the Markov Stability graph community detection algorithm in Python.
* [RMST](https://github.com/barahona-research-group/RMST) : Relaxed Minimum Spanning Tree: Computing the relaxed minimum spanning tree to sparsify networks whilst retaining dynamic structure.
* [StEP](https://github.com/barahona-research-group/StEP) :  Spatial-temporal Epidemiological Proximity: Characterising contact in disease outbreaks via a network model of spatial-temporal proximity.

