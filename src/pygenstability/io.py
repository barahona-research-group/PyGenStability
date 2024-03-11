"""I/O functions."""

import pickle


def save_results(all_results, filename="results.pkl"):
    """Save results in a pickle."""
    with open(filename, "wb") as results_file:
        pickle.dump(all_results, results_file)


def load_results(filename="results.pkl"):  # pragma: no cover
    """Load results from a pickle."""
    with open(filename, "rb") as results_file:
        return pickle.load(results_file)
