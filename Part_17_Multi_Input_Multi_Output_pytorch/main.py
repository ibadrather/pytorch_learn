import pandas as pd
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from models.get_model import get_model

try:
    os.system("clear")
except:
    pass

# Loading Data and Splitting into features and target arrays
data = pd.read_csv("mimo_data.csv")
features = data.values[ : , : 6]
targets = data.values[ : , 6 :]

# Train-Test Split
split_ratio = 0.8
features_train = features[ : int(len(features) * split_ratio)]
features_test = features[int(len(features) * split_ratio) : ]

targets_train = targets[ : int(len(features) * split_ratio)]
targets_test = targets[int(len(features) * split_ratio) : ]

# Number of input columns
n_inputs = features_train.shape[1]  # Number of features
n_outputs = targets_train.shape[1]  # Number of targets

print("Number of Inputs: ", n_inputs)
print("Number of Outputs: ", n_outputs)

#Define training hyperprameters.
batch_size = 50
num_epochs = 200
learning_rate = 0.001
size_hidden= 100
num_batches = len(features_train) // batch_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Assume that we are on a CUDA machine, then this should print a CUDA device:
print("Executing the model on :",device)

# Define the model
architecture = "simple"
model = get_model(architecture,n_features=n_inputs,  n_targets=n_outputs)
model.to(device)
summary(model, (1, n_inputs))

#Adam is a specific flavor of gradient decent which is typically better
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
criterion = torch.nn.MSELoss(reduction="sum")  # this is for regression mean squared loss

from sklearn.utils import shuffle
from torch.autograd import Variable
running_loss = 0.0
for epoch in range(num_epochs):
    # Mini batch learning
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        inputs = Variable(torch.FloatTensor(features_train[start:end])).to(device)
        outputs = Variable(torch.FloatTensor(targets_train[start:end])).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        #print("outputs",outputs)
        #print("outputs",outputs,outputs.shape,"labels",labels, labels.shape)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        
    print('Epoch {}'.format(epoch+1), "loss: ",running_loss)
    running_loss = 0.0