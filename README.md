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

