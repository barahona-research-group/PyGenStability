"""Test constructor module."""
from pathlib import Path
import numpy as np
from pygenstability import constructors
import yaml
from numpy.testing import assert_array_equal

from utils import graph

CONSTRUCTORS = [
    "linearized",
    "continuous_combinatorial",
    "continuous_normalized",
    "signed_modularity",
    "directed",
]
DATA = Path(__file__).absolute().parent / "data"


def _list_data(data):
    data = list(data)
    data[0] = np.array(data[0].toarray(), dtype=float).tolist()
    data[1] = np.array(data[1], dtype=float).tolist()
    data[2] = float(data[2]) if data[2] is not None else None
    return data


def test_load_constructor(graph):
    for constr in CONSTRUCTORS:
        data = _list_data(constructors.load_constructor(constr, graph).get_data(1))
        # yaml.dump(data, open(DATA / f"test_constructor_{constr}.yaml", "w"))
        expected_data = yaml.safe_load(open(DATA / f"test_constructor_{constr}.yaml", "r"))
        assert_array_equal(data[0], expected_data[0])
        assert_array_equal(data[1], expected_data[1])


def test_load_constructor_gap(graph):
    for constr in CONSTRUCTORS:
        data = _list_data(
            constructors.load_constructor(constr, graph, spectral_gap=True).get_data(1)
        )
        # yaml.dump(data, open(DATA / f"test_constructor_{constr}_gap.yaml", "w"))
        expected_data = yaml.safe_load(open(DATA / f"test_constructor_{constr}_gap.yaml", "r"))
        assert_array_equal(data[0], expected_data[0])
        assert_array_equal(data[1], expected_data[1])
