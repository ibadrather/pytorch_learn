import torch.nn as nn

class LSTMModel(nn.Module):
  def __init__(self, n_features, n_targets, n_hidden=128, n_layers=2):
    super().__init__()

    self.n_hidden = n_hidden

    self.lstm = nn.LSTM(
        input_size = n_features,
        hidden_size = n_hidden,
        batch_first = True,
        num_layers = n_layers, # Stack LSTMs
        dropout = 0.2
    )

    self.regressor = nn.Linear(n_hidden, n_targets)

  def forward(self, x):
    #self.lstm.flatten_parameters()  # For distrubuted training

    _, (hidden, _) = self.lstm(x)
    # We want the output from the last layer to go into the final
    # regressor linear layer
    out = hidden[-1] 

    return self.regressor(out)


if __name__ =="__main__":
    pass