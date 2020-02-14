"""i/o functions"""
import pickle


def save_all_results(all_results, filename="all_results.pkl"):
    """save results in a pickle"""
    pickle.dump(all_results, open(filename, "wb"))


def load_all_results(filename="all_results.pkl"):
    """load results from a pickle"""
    return pickle.load(open(filename, "rb"))
