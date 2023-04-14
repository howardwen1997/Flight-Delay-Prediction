import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

np.random.seed(2023)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, max_epochs):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.max_epochs = max_epochs

        # Loss per epoch placeholder
        self.train_loss_epoch = []
        self.train_err_epoch = []

        # lstm1, lstm2, linear are all layers in the network
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input):
        #print(input.shape)
        #print(input.view(6, 64 , -1).shape)
        lstm_out, _ = self.lstm(input)
        lstm_out = self.sigmoid(self.linear(lstm_out))
        return lstm_out

    def fit(self, train_loader, criterion, optimizer):
        # Epoch loop
        for i in range(1,self.max_epochs+1):
            train_loss = 0
            train_err_ct = 0

            # Mini batch loop
            for j, (features, target) in enumerate(train_loader):

                # Forward pass 
                output = self.forward(features)
                # get the loss
                loss = criterion(output, target)
                # print(j)
                # print(output)
                # print(loss)
                # print('\n')
                # calculate gradients 
                loss.backward()
                # gradient clipping\n",
                #torch.nn.utils.clip_grad_norm_(self.lstm.parameters(), max_norm=10, norm_type=2.0)\n",
                #update weights
                optimizer.step()
                # zero out the gradients
                optimizer.zero_grad() 

                # Track the loss and error rate
                train_loss += loss.item() * target.size(0)
                with torch.no_grad():
                    train_err_ct += sum((output.numpy().reshape(output.size(0)*6) > 0.5).astype(int) != target.numpy().reshape(output.size(0)*6))

            # calculate training loss per epoch
            train_loss_mean = train_loss / len(train_loader.dataset) / 6
            self.train_loss_epoch.append(train_loss_mean)
            train_err = train_err_ct / len(train_loader.dataset) / 6
            self.train_err_epoch.append(train_err)

            # Print/return training loss and error rate in each epoch
            print(f'Epoch {i}')
            print(f'Train Loss: {train_loss_mean:.4f}')
            print(f'Train Error Rate: {train_err*100:.4f}%\n')

            # Stop training before maximum number of epochs is reached if:
                # more than 10 epochs have been run
                # difference in loss from previous epoch < 1e-10
            # if i > 10:
            #     if np.abs(self.train_loss_epoch[-1] - self.train_loss_epoch[-2]) < 1e-5:
            #         break

    def predict(self, test_loader, criterion):
        test_loss = 0
        test_err_ct = 0

        with torch.no_grad(): # no backprop step so turn off gradients
    
            for j,(features,target) in enumerate(test_loader):

                # Compute prediction output and loss
                output = self.forward(features)
                loss = criterion(output, target)

                # Measure loss and error rate and record
                test_loss += loss.item() * target.size(0)
                test_err_ct += sum((output.numpy().reshape(output.size(0)*6) > 0.5).astype(int) != target.numpy().reshape(output.size(0)*6))

        # Print/return test loss and error rate
        test_loss_mean = test_loss / len(test_loader.dataset) / 6
        test_err = test_err_ct / len(test_loader.dataset) / 6
        print(f'Test Loss: {test_loss_mean:.3f}')
        print(f'Test Error Rate: {test_err*100:.4f}%\\n')

        return round(test_loss_mean,3), str(round(test_err*100,3)) + "%"                