import torch
import torch.nn.functional as F
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, n_features, n_targets, hiddenA=100, hiddenB=50):
        super(MLPModel, self).__init__()
        self.linearA = nn.Linear(n_features, hiddenA)
        self.linearB = nn.Linear(hiddenA, hiddenB)
        self.linearC = nn.Linear(hiddenB, n_targets)

    def forward(self, x):
        yA = F.relu(self.linearA(x))
        yB = F.relu(self.linearB(yA))
        return self.linearC(yB)


if __name__ =="__main__":
    pass