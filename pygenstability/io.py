"""i/o functions"""
import pickle
import yaml


def load_params(params_path):
    """load params fro yaml file"""
    with open(params_path, "r") as params_file:
        try:
            return yaml.safe_load(params_file)
        except yaml.YAMLError:
            raise Exception("Could not read yaml params file {}".format(params_path))


def save_results(all_results, filename="results.pkl"):
    """save results in a pickle"""
    with open(filename, "wb") as results_file:
        pickle.dump(all_results, results_file)


def load_results(filename="results.pkl"):
    """load results from a pickle"""
    with open(filename, "rb") as results_file:
        return pickle.load(results_file)
