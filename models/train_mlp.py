import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from FlightDataset import FlightDataset
from FlightMLP import FlightMLP
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

# hyperparams
#######################
batchsize = 16
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

# Training params
#######################
learning_rate = 1e-4
max_epochs = 500
pos_weight = 3 # 3x as many not-delayed flights
patience_time = 20 # early stopping
#######################

# model architecture
layer_set = [
    [30],
    [50],
    [100],
    [200],
    [20, 20],
    [50, 50],
    [100, 100],
    [20, 20, 20],
    [50, 50, 50],
    [100, 100, 100],
    [200, 200, 200],
]

learning_rates = [1e-3, 1e-4, 1e-5]
             
dropouts=[False, True]

for layers in layer_set:
    for dropout in dropouts:
        for lr in learning_rates:
            
            print(f'training layers={layers} dropout={dropout} lr={lr}')

            save_name = 'layers'
            for layer in layers:
                save_name +=  '_' + str(layer)
            if dropout:
                save_name += '_drop'
            save_name += '_lr_' + str(lr)

            model_name ='weights/' + save_name + '.pt'
            loss_filename = '../results/loss' + save_name + '.csv'
            plot_filename = '../plots/' + save_name + '.png'

            ##############################

            mlp = FlightMLP(layers, dropout=dropout)
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight), reduction='sum')
            optimizer = torch.optim.SGD(params=mlp.parameters(), lr=learning_rate)

            epoch_list = np.arange(1, max_epochs+1)
            train_losses = np.zeros(max_epochs)
            valid_losses = np.zeros(max_epochs)
            weights = [] # save weights

            for epoch in range(max_epochs):
                
                correct = 0
                mlp.train()
                
                for i, batch in enumerate(iter(train_dataloader)):
                    X = batch['features'].float()
                    labels = batch['delayed'].flatten()
                    
                    logit_pred = mlp.forward(X).flatten()
                    labels_pred = torch.sigmoid(logit_pred).round()
                    
                    optimizer.zero_grad()
                    loss = criterion(logit_pred, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_losses[epoch] += loss.item()
                    correct += (labels_pred == labels).sum()
            
                train_losses[epoch] = train_losses[epoch]/n_train 
                
                with torch.no_grad():
                    mlp.eval()
                    valid_labels = torch.tensor(validation_dataset.data[:,-1])
                    valid_pred_logit = mlp(torch.tensor(validation_dataset.data[:, :-1]).float()).round().reshape(-1)
                    valid_loss = criterion(valid_pred_logit, valid_labels)
                    valid_losses[epoch] = valid_loss.item()/n_validation
                
                # save weights
                weights.append(mlp.state_dict())
                weights = weights[-patience_time:]

                if epoch > patience_time:
                    if valid_losses[epoch] > valid_losses[epoch - patience_time]:
                        print('Stopping early\n')
                        torch.save(weights[0], model_name)
                        break
                
                print(f'Epoch {epoch_list[epoch]}: loss: {train_losses[epoch]:>7f} accuracy: {correct/n_train:>7f} Val loss: {valid_losses[epoch]:.7f}')
                
                torch.save(weights[0], model_name)

            # test metrics
            mlp.eval()
            test_label = test_dataset.data[:, -1]
            test_pred = torch.sigmoid(mlp(torch.tensor(test_dataset.data[:, :-1]).float())).round().detach().numpy()

            precision, recall, f1, support = precision_recall_fscore_support(test_label, test_pred)
            accuracy = accuracy_score(test_label, test_pred)

            print('precision: ', precision[1])
            print('recall: ', recall[1])
            print('f1: ', f1[1])
            print('accuracy', accuracy)

            # save values
            train_losses = train_losses[:epoch+1]
            valid_losses = valid_losses[:epoch+1]
            epoch_list = epoch_list[:epoch+1]

            df = pd.DataFrame(
                {'Epoch' : epoch_list,
                'Train_loss' :  train_losses,
                'val_loss' : valid_losses
                }
            )
            df.to_csv(loss_filename, index=False)

            # plot loss
            fig, ax = plt.subplots()
            ax.plot(epoch_list, train_losses, label='training loss')
            ax.plot(epoch_list, valid_losses, label='validation loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Avg. Loss')
            ax.legend()
            fig.savefig(plot_filename)
            fig.clear()
            
            print("---------------------\n")
            