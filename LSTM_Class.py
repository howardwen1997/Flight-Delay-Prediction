import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

np.random.seed(2023)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        # lstm1, lstm2, linear are all layers in the network
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

        self.train_loss_epoch = []
        self.train_err_epoch = []
        self.val_loss_epoch = []
        self.val_err_epoch = [] 
        
    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        lstm_out = self.linear(lstm_out)
        return lstm_out

    def fit(self, data_loader, criterion, optimizer):
        train_loss = 0
        train_err_ct = 0
        
        self.train_output_prob_array = []
        self.train_output_binary_array = []
        self.train_target_array = []       
        # Mini batch loop
        for j, (features, target) in enumerate(data_loader):
            # Forward pass 
            output = self.forward(features)
            # get the loss
            loss = criterion(output, target)
            # calculate gradients 
            loss.backward()
            #update weights
            optimizer.step()
            # zero out the gradients
            optimizer.zero_grad() 

            # Track the loss and error rate
            train_loss += loss.item() * target.size(0) * 6
            with torch.no_grad():
                train_output_prob_array = output.numpy().reshape(output.size(0)*6)
                self.train_output_prob_array.extend(list(train_output_prob_array))           
                train_output_binary_array = (train_output_prob_array > 0.5).astype(int)
                self.train_output_binary_array.extend(list(train_output_binary_array))
                
                train_target_array = target.numpy().reshape(output.size(0)*6)
                self.train_target_array.extend(list(train_target_array))
                train_err_ct += sum(train_output_binary_array != train_target_array)

        # calculate training loss per epoch
        train_loss_mean = train_loss / len(data_loader.dataset) / 6
        self.train_loss_epoch.append(train_loss_mean)
        train_err = train_err_ct / len(data_loader.dataset) / 6
        self.train_err_epoch.append(train_err)

        # Print/return training loss and error rate in each epoch
        print(f'Train Loss: {train_loss_mean:.4f}')
        print(f'Train Accuracy Rate: {(1-train_err)*100:.4f}%')
        f1_metrics = precision_recall_fscore_support(self.train_target_array, self.train_output_binary_array, average='binary')
        print(f'Precision: {f1_metrics[0]:.4f}')
        print(f'Recall: {f1_metrics[1]:.4f}')
        print(f'F1: {f1_metrics[2]:.4f}\n')
        #print(f'Train AUC: {roc_auc_score(self.train_target_array, self.train_output_prob_array):.4f}')

    def predict(self, data_loader, criterion, set):
        val_loss = 0
        val_err_ct = 0

        self.val_output_prob_array = []
        self.val_output_binary_array = []
        self.val_target_array = []

        test_loss = 0
        test_err_ct = 0
        self.test_output_prob_array = []
        self.test_output_binary_array = []
        self.test_target_array = []

        with torch.no_grad(): # no backprop step so turn off gradients
            for j,(features,target) in enumerate(data_loader):
                # Compute prediction output and loss
                output = self.forward(features)
                loss = criterion(output, target)
                
                # Measure loss and error rate and record
                if set == 'validation':
                    val_loss += loss.item() * target.size(0) * 6
                    val_output_prob_array = output.numpy().reshape(output.size(0)*6)
                    self.val_output_prob_array.extend(list(val_output_prob_array))           
                    val_output_binary_array = (val_output_prob_array > 0.5).astype(int)
                    self.val_output_binary_array.extend(list(val_output_binary_array))
                    
                    val_target_array = target.numpy().reshape(output.size(0)*6)
                    self.val_target_array.extend(list(val_target_array))
                    val_err_ct += sum(val_output_binary_array != val_target_array)

                if set == 'test':
                    test_loss += loss.item() * target.size(0) * 6
                    test_output_prob_array = output.numpy().reshape(output.size(0)*6)
                    self.test_output_prob_array.extend(list(test_output_prob_array))           
                    test_output_binary_array = (test_output_prob_array > 0.5).astype(int)
                    self.test_output_binary_array.extend(list(test_output_binary_array))
                    
                    test_target_array = target.numpy().reshape(output.size(0)*6)
                    self.test_target_array.extend(list(test_target_array))
                    test_err_ct += sum(test_output_binary_array != test_target_array)

        if set == 'validation':
            val_loss_mean = val_loss / len(data_loader.dataset) / 6
            self.val_loss_epoch.append(val_loss_mean)
            val_err = val_err_ct / len(data_loader.dataset) / 6
            self.val_err_epoch.append(val_err)
            print(f'Validation Loss: {val_loss_mean:.6f}')
            print(f'Validation Accuracy Rate: {(1-val_err)*100:.4f}%')
            f1_metrics = precision_recall_fscore_support(self.val_target_array, self.val_output_binary_array, average='binary')
            print(f'Validation Precision: {f1_metrics[0]:.4f}')
            print(f'Validation Recall: {f1_metrics[1]:.4f}')
            print(f'Validation F1: {f1_metrics[2]:.4f}\n')          

        if set == 'test':
            test_loss_mean = test_loss / len(data_loader.dataset) / 6
            test_err = test_err_ct / len(data_loader.dataset) / 6
            print(f'Test Loss: {test_loss_mean:.6f}')
            print(f'Test Accuracy Rate: {(1-test_err)*100:.6f}%')
            f1_metrics = precision_recall_fscore_support(self.test_target_array, self.test_output_binary_array, average='binary')
            print(f'Test Precision: {f1_metrics[0]:.6f}')
            print(f'Test Recall: {f1_metrics[1]:.6f}')
            print(f'Test F1: {f1_metrics[2]:.6f}')
            print(f'Test AUC: {roc_auc_score(self.test_target_array, self.test_output_prob_array):.6f}\n')

            return f'Accuracy: {(1-test_err)*100:.6f}%, Precision: {f1_metrics[0]:.6f}, Recall: {f1_metrics[1]:.6f}, F1: {f1_metrics[2]:.6f}, AUC: {roc_auc_score(self.test_target_array, self.test_output_prob_array):.6f}'