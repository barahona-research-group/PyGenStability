"""Setup for the pybind11 C++ extension.

All project metadata lives in pyproject.toml (PEP 621). This file
only declares the optional C++ extension.
"""

from pybind11.setup_helpers import Pybind11Extension
from setuptools import setup

setup(
    ext_modules=[
        Pybind11Extension(
            "pygenstability.generalized_louvain",
            ["src/pygenstability/generalized_louvain/generalized_louvain.cpp"],
            include_dirs=["extra", "generalizedLouvain"],
            cxx_std=11,
            # use lemon header-only (its compiled library, e.g. lemon::INVALID, is not built)
            define_macros=[("LEMON_ONLY_TEMPLATES", None)],
            optional=True,
        ),
    ],
)
