# PyGenStability
Python wrapper of the generalised Louvain code of Micheal Schaub at https://github.com/michaelschaub/generalizedLouvain with python code to run various versions of MarkovStability. 

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

Example
-------

In the `test` folder, a demo script with stochastic block model can be tried with 

```
python test_cpp.py
```



