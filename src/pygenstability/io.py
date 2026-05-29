"""I/O functions."""

from __future__ import annotations

import pickle


def save_results(all_results: dict, filename: str = "results.pkl") -> None:
    """Save results in a pickle."""
    with open(filename, "wb") as results_file:
        pickle.dump(all_results, results_file)


def load_results(filename: str = "results.pkl") -> dict:  # pragma: no cover
    """Load results from a pickle."""
    with open(filename, "rb") as results_file:
        return pickle.load(results_file)
