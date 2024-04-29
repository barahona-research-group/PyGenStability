"""Utils for tests."""
import numpy as np
from copy import deepcopy


def _to_list(_data):
    """Convert dict to list with floats for yaml encoding."""
    data = deepcopy(_data)
    data.pop("NVI", None)  # NVI computation is unstable, we don't test it
    data.pop("ttprime", None)  # ttprime computation is unstable, we don't test it
    for key, val in data.items():
        if isinstance(val, dict):
            data[key] = _to_list(data[key])
        if isinstance(val, (np.ndarray, list)):
            data[key] = np.array(val, dtype=float).tolist()
    return data
