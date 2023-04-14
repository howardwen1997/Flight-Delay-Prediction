import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from FlightDataset import FlightDataset
from FlightMLP import FlightMLP
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from sklearn.model_selection import train_test_split

# hyperparams
#######################
batchsize = 16
learning_rate = 0.001
max_epochs = 20
pos_weight = 1
#######################

# get data
data = np.load('data/airline_final.npy')
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
data = None # free up memory

train_dataset = FlightDataset(train_data)
train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
train_data = None

test_dataset = FlightDataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)
test_data = None

n_train = train_dataset.__len__()
n_test = test_dataset.__len__()

mlp = FlightMLP(learning_rate=1e-3, max_epochs=10)

criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight), reduction='sum')
optimizer = torch.optim.Adam(lr=learning_rate, params=mlp.parameters())

for epoch in range(max_epochs):
    lossval = 0
    correct = 0
    for i, batch in enumerate(iter(train_dataloader)):
        X = batch['features'].float()
        labels = batch['delayed'].flatten()
        
        logit_pred = mlp.forward(X).flatten()
        labels_pred = torch.sigmoid(logit_pred).round()
        
        optimizer.zero_grad()
        loss = criterion(logit_pred, labels)
        loss.backward()
        optimizer.step()
        
        lossval += loss.item()
        correct += (labels_pred == labels).sum()
    
    print(f'Epoch {epoch}: loss {lossval/n_train} accuracy {correct/n_train}')

# test metrics
test_label = test_dataset.data[:, -1]
test_pred = torch.sigmoid(mlp(torch.tensor(test_dataset.data[:, :-1]).float())).round().detach().numpy()
print(precision_recall_fscore_support(test_label, test_pred))