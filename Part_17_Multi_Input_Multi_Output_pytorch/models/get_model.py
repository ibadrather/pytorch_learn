from tkinter import N
from .simple_net import SimpleNet
from.lstm import LSTMModel
from .mlp import MLPModel
from .fc_net import FCNetwork

def get_model(arch, n_features, n_targets):
    if arch == "simple":
        network = SimpleNet(n_features=n_features, n_targets=n_targets)
    
    elif arch == "lstm":
        network = LSTMModel(n_features=n_features, n_targets=n_targets)

    elif arch == "mlp":
        network = MLPModel(n_features=n_features, n_targets=n_targets)

    elif arch == "fc_net":
        network = FCNetwork(n_features=n_features, n_targets=n_targets, 
                hidden_layers=[1024, 512, 256, 128, 64], drop_p=0.5)

    else:
        raise ValueError("Invalid architecture: ", arch)
    return network

if __name__ =="__main__":
    pass