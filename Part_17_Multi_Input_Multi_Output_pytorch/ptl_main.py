import pandas as pd
import os
import pickle

import torch
from torchsummary import summary
from models.get_model import get_model
from pytorch_lightning.callbacks.progress import TQDMProgressBar
import  pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from ptl_dataloader import MIMODataModule
from ptl_model import MIMOPredictor


try:
    os.system("clear")
except:
    pass

# Random Seed Pytorch Lightning
pl.seed_everything(42)

# Loading Data and Splitting into features and target arrays
#data = pd.read_csv("mimo_data.csv")
with open('mimo_data.pkl', 'rb') as f:
    data = pickle.load(f)

features = data.values[ : , : 6]
targets = data.values[ : , 6 :]

# Train-Test Split
split_ratio = 0.8
train_features = features[ : int(len(features) * split_ratio)]
test_features = features[int(len(features) * split_ratio) : ]

train_targets = targets[ : int(len(features) * split_ratio)]
test_targets = targets[int(len(features) * split_ratio) : ]

print("Train Size: ", len(train_features))
print("Tes Size: ", len(test_features))

# Number of input columns
n_inputs = train_features.shape[1]  # Number of features
n_outputs = train_targets.shape[1]  # Number of targets

print("Number of Inputs: ", n_inputs)
print("Number of Outputs: ", n_outputs)

# Define the model
architecture = "fc_net"
net = get_model(architecture,n_features=n_inputs,  n_targets=n_outputs)
summary(net, (1, n_inputs))

N_EPOCHS = 8
BATCH_SIZE = 10

data_module = MIMODataModule(train_features, train_targets, 
      test_features, test_targets, batch_size=BATCH_SIZE)
  
data_module.setup()


# Model
model = MIMOPredictor(net)


# Defining Callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k = 2,
    verbose = True,
    monitor = "val_loss",
    mode = "min"
)

# Log to Tensor Board
logger = TensorBoardLogger("lightning_logs", name = "mimo-predict")

# Stop trainining if model is not improving
early_stopping_callback = EarlyStopping(monitor = "val_loss", patience = 30)

# Progress bar
progress_bar = TQDMProgressBar(refresh_rate=5)
 

N_EPOCHS = 1
trainer = pl.Trainer(
    logger = logger,
    enable_progress_bar=True,
    callbacks = [early_stopping_callback, early_stopping_callback, progress_bar],
    max_epochs = N_EPOCHS,
    gpus = 1,
    )

trainer.fit(model, data_module)
