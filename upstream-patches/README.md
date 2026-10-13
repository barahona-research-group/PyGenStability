# Upstream patches (not part of PyGenStability)

Saved here so they are not lost; apply each set in its own repository with `git am`.

## leidenalg (vtraag/leidenalg, base 9890a99)
- `leidenalg/0001-*`: pass `-DCMAKE_BUILD_TYPE=Release` in `scripts/build_libleidenalg.sh`.
  The Linux/macOS wheels currently build libleidenalg at -O0 (the shipped .so exports
  out-of-line `std::vector::operator[]`); an optimised build is 4-5x faster, identical results
  (20k-node CPM: 2.2-2.6 s -> 0.5 s per optimise call).

## libleidenalg (vtraag/libleidenalg, base fe2d4e7)
Full stack `0001`-`0005` (apply in order). The same changes split into three independent PRs,
each based on fe2d4e7:
- `branch-release-build/`: default `CMAKE_BUILD_TYPE` to Release (the 4-5x above).
- `branch-refinement-alloc-bytevectors/`: drop the unused O(n_communities) allocation per call
  in constrained `get_neigh_comms` (O(n^2) at the first refinement level) and use
  `vector<char>` instead of `vector<bool>` in hot loops. Identical results.
- `branch-csr-zero-weight-layers/`: eager CSR adjacency instead of lazy igraph lists copied on
  every call (~-20% instructions; changes `Graph::get_neighbours`/`get_neighbour_edges` to
  return an `IndexRange`, an exported-API change) and skip edge loops for all-zero-weight
  layers (multiplex null layers, ~-30% on the 2-layer case).

All five together on -O3: 20k-node CPM single layer ~0.44 s, 2-layer multiplex ~0.65 s
(wheel: 2.2-2.6 s / 3.9-4.2 s); memberships and qualities bit-identical on 180 equivalence
cases and the 20k workload; leidenalg's 688 tests pass.

## generalizedLouvain (michaelschaub/generalizedLouvain, base 8c9f1f2)
- `generalizedLouvain/0001-*`: insert reduced-graph edges in sorted order (cross-platform
  determinism; fixes the 3 Windows `test_dataclustering` failures, see #119/#120). Also open as
  https://github.com/arnaudon/generalizedLouvain/pull/1.
