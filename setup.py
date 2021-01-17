from setuptools import setup, find_packages

try:
    from pybind11.setup_helpers import Pybind11Extension
except ImportError:
    from setuptools import Extension as Pybind11Extension
import sys
import setuptools

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "pygenstability.generalized_louvain",
        ["pygenstability/generalized_louvain/generalized_louvain.cpp"],
        include_dirs=["extra"],
    ),
]

setup(
    name="pygenstability",
    version=__version__,
    author="Alexis Arnaudon",
    author_email="alexis.arnaudon@epfl.ch",
    url="https://github.com/ImperialCollegeLondon/PyGenStability",
    description="Python binding of generalised Markov Stability",
    ext_modules=ext_modules,
    setup_requires=["pybind11>=2.6.0"],
    zip_safe=False,
    install_requires=[
        "numpy>=1.18.1",
        "scipy>=1.4.1",
        "matplotlib>=3.1.3",
        "networkx>=2.4",
        "sklearn>=0.0",
        "cmake>=3.16.3",
        "pyyaml>=5.3",
        "click>=7.0",
        "tqdm>=4.45.0",
    ],
    extras_require={"plotly": ["plotly>=3.6.0"]},
    entry_points={"console_scripts": ["pygenstability=pygenstability.app:cli"]},
    packages=find_packages(),
)
