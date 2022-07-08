import torch.nn as nn
import torch.nn.functional as F


class FCNetwork(nn.Module):
    def __init__(self, n_features, n_targets, 
                hidden_layers=[1024, 512, 256, 128, 64],
                drop_p=0.5):

        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            n_features: integer, size of the input layer
            n_targets: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''

        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(n_features, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.dropout = nn.Dropout(p=drop_p)

        self.output = nn.Linear(hidden_layers[-1], n_targets)
        
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return x
    
    
if __name__ =="__main__":
    pass