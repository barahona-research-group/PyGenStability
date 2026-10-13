"""Check that the compiled Louvain extension is present and works (used by cibuildwheel)."""

import numpy as np

from pygenstability import generalized_louvain

# two triangles joined by one edge, modularity-like null model
rows = np.array([1, 2, 2, 4, 5, 5, 3])
cols = np.array([0, 0, 1, 3, 3, 4, 2])
weights = np.ones(len(rows)) / 7
null_model = np.full((2, 6), 1 / 6)
stability, partition = generalized_louvain.run_louvain(
    rows, cols, weights, len(weights), null_model, 2, 1.0, 0
)
assert len(partition) == 6, partition
assert partition[0] == partition[1] == partition[2] != partition[3] == partition[4] == partition[5]
print("generalized_louvain OK:", stability, partition)
