import torch
import torch.nn.functional as F

class SimpleNet(torch.nn.Module):
    def __init__(self, n_features,  n_targets, size_hidden=100):
        super(SimpleNet, self).__init__()
        self.hidden = torch.nn.Linear(n_features, size_hidden)   # hidden layer
        self.predict = torch.nn.Linear(size_hidden, n_targets)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

if __name__ =="__main__":
    pass
