# PyGenStability
Python wrapper of the generalised Louvain, including Markov Stability


Based on the cppyy python packaging of https://github.com/camillescott/cppyy-bbhash
To install: 

    conda create -n PyGenStability python=3 cmake cxx-compiler c-compiler clangdev libcxx libstdcxx-ng libgcc-ng pytest -c conda-forge
    conda activate PyGenStability 
    pip install cppyy clang

    mkdir build; cd build
    cmake ..
    make

here is does not work, looking for lemon library


    python setup.py bdist_wheel
    pip install dist/cppyy_bbhash-*.whl
