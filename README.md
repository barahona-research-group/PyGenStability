# PyGenStability
Python wrapper of the generalised Louvain code of https://github.com/michaelschaub/generalizedLouvain with python code to run various versions of MarkovStability. 


Based on the cppyy python packaging of https://github.com/camillescott/cppyy-bbhash
To install the louvain_to_python wrapper, go to Louvain_to_python subfolder and run: 

    conda install cmake cxx-compiler c-compiler clangdev libcxx libstdcxx-ng libgcc-ng pytest lemon -c conda-forge
    pip install cppyy clang

    mkdir build; cd build
    cmake ..
    make
    make install

Then in the main folder, install the python code with:

    python setup.py install

An example notebook is in the folder test.
