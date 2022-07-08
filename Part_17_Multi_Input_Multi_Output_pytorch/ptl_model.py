import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim

class MIMOPredictor(pl.LightningModule):
  def __init__(self, model):
    super().__init__()
    self.model = model
    self.criterion = nn.MSELoss()

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch, batch_idx):
    features, targets = batch
    
    # Putting data into the network
    output = self.forward(features)
    # for lstm
    if output.shape != targets.shape:
      output = output.unsqueeze()

    # Calculating Loss
    loss = self.criterion(output, targets)

    self.log("train_loss", loss, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    features, targets = batch

    # Putting data into the network
    output = self.forward(features)

    # for lstm
    if output.shape != targets.shape:
      output = output.unsqueeze()

    # Calculating Loss
    loss = self.criterion(output, targets)

    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss
  
  def test_step(self, batch, batch_idx):
    features, targets = batch

    # Putting data into the network
    output = self.forward(features)

    # for lstm
    if output.shape != targets.shape:
      output = output.unsqueeze()

    # Calculating Loss
    loss = self.criterion(output, targets)

    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def configure_optimizers(self):
    return optim.Adam(self.model.parameters(), lr=1e-4)


if __name__ =="__main__":
    pass
  