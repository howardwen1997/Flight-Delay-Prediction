import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from FlightDataset import FlightDataset
from FlightMLP import FlightMLP
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# hyperparams
#######################
batchsize = 16
learning_rate = 1e-3
max_epochs = 43
pos_weight = 3 # 3x as many not-delayed flights
patience_time = 10 # every 10 epochs check loss on validation set, early stopping if loss increases
#######################

# get data
#######################
data = np.load('../data/airline_final.npy')
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, validation_data = train_test_split(train_data, test_size=.15, random_state=42)
data = None # free up memory

train_dataset = FlightDataset(train_data)
train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
train_data = None

validation_dataset = FlightDataset(validation_data)
validation_dataloader = DataLoader(validation_dataset, batch_size=batchsize, shuffle=True)
validation_data = None

test_dataset = FlightDataset(test_data)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True)
test_data = None

n_train = train_dataset.__len__()
n_validation = validation_dataset.__len__()
n_test = test_dataset.__len__()
#######################

mlp = FlightMLP()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight), reduction='sum')
optimizer = torch.optim.SGD(params=mlp.parameters(), lr=learning_rate)

epoch_list = np.arange(1, max_epochs+1)
losses = np.zeros(max_epochs)
previous_valid_accuracy = 0
for epoch in range(max_epochs):
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
        
        losses[epoch] += loss.item()
        correct += (labels_pred == labels).sum()
   
    losses[epoch] = losses[epoch]/n_train 
    print(f'Epoch {epoch_list[epoch]}: loss {losses[epoch]} accuracy {correct/n_train:>7f}')
    
    if epoch % patience_time == 0:
        with torch.no_grad():
            valid_label = validation_dataset.data[:,-1]
            valid_pred = torch.sigmoid(mlp(torch.tensor(validation_dataset.data[:, :-1]).float())).round().detach().numpy()
            valid_acc = accuracy_score(valid_label, valid_pred)
            print(f"VALIDATION ACCURACY: {valid_acc}")
            print(f"previous: {previous_valid_accuracy}")
            if valid_acc < previous_valid_accuracy:
                break
            previous_valid_accuracy = valid_acc

    # print test set metrics
    if epoch % 25 == 0:
        with torch.no_grad():
            test_label = test_dataset.data[:, -1]
            test_pred = torch.sigmoid(mlp(torch.tensor(test_dataset.data[:, :-1]).float())).round().detach().numpy()

            precision, recall, f1, support = precision_recall_fscore_support(test_label, test_pred)
            accuracy = accuracy_score(test_label, test_pred)

            print('precision: ', precision[1])
            print('recall: ', recall[1])
            print('f1: ', f1[1])
            print('accuracy', accuracy)
        

# test metrics
test_label = test_dataset.data[:, -1]
test_pred = torch.sigmoid(mlp(torch.tensor(test_dataset.data[:, :-1]).float())).round().detach().numpy()

precision, recall, f1, support = precision_recall_fscore_support(test_label, test_pred)
accuracy = accuracy_score(test_label, test_pred)

print('precision: ', precision[1])
print('recall: ', recall[1])
print('f1: ', f1[1])
print('accuracy', accuracy)

# plot loss
losses = losses[:epoch+1]
# epochs_list = epoch_list[:epoch+1]

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Avg. Training Loss')
plt.show()
# plt.savefig('plots/mlp_loss.png')