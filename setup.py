from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension

__version__ = "0.0.2"

ext_modules = [
    Pybind11Extension(
        "pygenstability.generalized_louvain",
        ["pygenstability/generalized_louvain/generalized_louvain.cpp"],
        include_dirs=["extra"],
        extra_compile_args=["-std=c++11"],
    ),
]

setup(
    name="pygenstability",
    version=__version__,
    author="Alexis Arnaudon",
    author_email="alexis.arnaudon@epfl.ch",
    url="https://github.com/ImperialCollegeLondon/PyGenStability",
    description="Python binding of generalised Louvain with Markov Stability",
    ext_modules=ext_modules,
    setup_requires=["pybind11>=2.6.0"],
    install_requires=[
        "numpy>=1.18.1",
        "scipy>=1.4.1",
        "matplotlib>=3.1.3",
        "networkx>=2.4",
        "sklearn>=0.0",
        "cmake>=3.16.3",
        "click>=7.0",
        "tqdm>=4.45.0",
        "pybind11>=2.6.2",
        "pandas>=1.0.0",
    ],
    zip_safe=False,
    extras_require={"plotly": ["plotly>=3.6.0"]},
    entry_points={"console_scripts": ["pygenstability=pygenstability.app:cli"]},
    packages=find_packages(),
    include_package_data=True,
)
