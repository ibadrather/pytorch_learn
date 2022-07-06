from tkinter import N
from .simple_net import SimpleNet
from.lstm import LSTMModel
from .mlp import MLPModel

def get_model(arch, n_features, n_targets):
    if arch == "simple":
        network = SimpleNet(n_features=n_features, n_targets=n_targets)
    
    elif arch == "lstm":
        network = LSTMModel(n_features=n_features, n_targets=n_targets)

    elif arch == "mlp":
        network = MLPModel(n_features=n_features, n_targets=n_targets)

    else:
        raise ValueError("Invalid architecture: ", arch)
    return network