import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim

class MIMOPredictor(pl.LightningModule):
  def __init__(self, model):
    super().__init__()
    self.model = model
    self.criterion = nn.MSELoss()

  def forward(self, x, labels=None):
    output = self.model(x)
    loss = 0

    if labels is not None:
      loss = self.criterion(output, labels.unsqueeze(dim=1))
    return loss, output

  def training_step(self, batch, batch_idx):
    features, targets = batch
    # print("Feature Shape", features.shape)
    print("Target Shape", targets.shape)

    loss, output = self.forward(features, targets)

    self.log("train_loss", loss, prog_bar=True, logger=True)
    return loss

  def validation_step(self, batch, batch_idx):
    features, targets = batch

    loss, output = self.forward(features, targets)

    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss
  
  def test_step(self, batch, batch_idx):
    features, targets = batch

    loss, output = self.forward(features, targets)

    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def configure_optimizers(self):
    return optim.Adam(self.model.parameters(), lr=1e-4)


if __name__ =="__main__":
    pass