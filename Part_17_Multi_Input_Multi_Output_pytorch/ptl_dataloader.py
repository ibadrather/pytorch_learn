import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count
import torch

class MIMODataset(Dataset):
  def __init__(self, features, targets):
    self.features = features
    self.targets = targets

  def __len__(self):
    return len(self.features)
  
  def __getitem__(self, idx):
    feature = self.features[idx]
    target = self.targets[idx]

    return torch.Tensor(feature), torch.Tensor(target)


class MIMODataModule(pl.LightningDataModule):
  def __init__(
      self, train_features, train_targets, 
      test_features, test_targets, 
      batch_size = 1
  ):
    super().__init__()
    self.train_features = train_features
    self.train_targets = train_targets
    self.test_features = test_features
    self.test_targets = test_targets
    self.batch_size = batch_size
  
  def setup(self, stage=None):
    self.train_dataset = MIMODataset(self.train_features, self. train_targets)
    self.test_dataset = MIMODataset(self.test_features, self. test_targets)
  
  def train_dataloader(self):
    return DataLoader(
        self.train_dataset,
        batch_size = self.batch_size,
        shuffle = False,
        num_workers = cpu_count()
    )

  def val_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size = 1, #self.batch_size
        shuffle = False,
        num_workers = cpu_count()
    )

  def test_dataloader(self):
    return DataLoader(
        self.test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = cpu_count()
    )

if __name__ =="__main__":
    pass