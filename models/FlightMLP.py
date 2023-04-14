import numpy as np

import torch
import torch.nn as nn

'''
 Input: 1-channel image of size 28x28 pixels
 Fully-connected linear layer 1: input image with bias; output with 128 nodes
 ReLU activation function
 Fully-connected linear layer 2: input from layer 1 with bias; output with 10 nodes
 Softmax activation function on output layer (note, you do not need to specify this
anywhere, it is included when using cross entropy loss torch.nn.CrossEntropyLoss()).
'''

# Fully connected neural network with one hidden layer
class FlightMLP(nn.Module):
    
    def __init__(self, learning_rate, max_epochs):
        '''
        input_size: [int], feature dimension 
        learning_rate: learning rate for gradient descent,
        max_epochs: maximum number of epochs to run gradient descent
        '''
        ### Construct your MLP Here (consider the recommmended functions in homework writeup)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.epsilon = 1e-3
        
        super(FlightMLP, self).__init__()
        
        # sizes (arbitrar)
        input_size = 31
        hl1 = 50
        hl2 = 50

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hl1),
            nn.ReLU(),
            nn.Linear(hl1, hl2),
            nn.ReLU(),
            nn.Linear(hl2, 1),
        )


    def forward(self, x):
        ''' Function to do the forward pass with images x '''
        ### Use the layers you constructed in __init__ and pass x through the network
        ### and return the output
        return self.mlp(x)

    # def fit(self, train_loader, criterion, optimizer):
    #     '''
    #     Function used to train the MLP

    #     train_loader: includes the feature matrix and class labels corresponding to the training set,
    #     criterion: the loss function used,
    #     optimizer: which optimization method to train the model.
    #     '''
    #     optimizer = optimizer(self.mlp.parameters(), lr = self.learning_rate)
    #     avg_loss = np.zeros(self.max_epochs)
    #     error_rate = np.zeros(self.max_epochs)
    #     prev_loss = np.infty
        
    #     # Epoch loop
    #     for i in range(self.max_epochs):            

    #         # Mini batch loop
    #         correct = 0
    #         total_loss = 0
    #         for j,(images,labels) in enumerate(train_loader):

    #             # Forward pass (consider the recommmended functions in homework writeup)
                
    #             labels_pred = self.forward(images)
    #             labels_max = torch.argmax(labels_pred, axis=1)
            
    #             # Backward pass and optimize (consider the recommmended functions in homework writeup)
    #             # Make sure to zero out the gradients using optimizer.zero_grad() in each loop
    #             optimizer.zero_grad()
    #             loss = criterion(labels_pred, labels)
    #             loss.backward()
    #             optimizer.step()
                            
    #             # Track the loss and error rate
    #             m = images.size()[0]
    #             correct += (labels_max == labels).sum()
    #             total_loss += loss.item() * m
                
    #         # Print/return training loss and error rate in each epoch
    #         n = len(train_loader.dataset)
    #         avg_loss[i] = total_loss/n
    #         error_rate[i]= 1.0 - correct/n
            
    #         print(f"epoch: {i+1} loss: {avg_loss[i]:>7f} error rate: {error_rate[i]:>7f}")
            
    #         # check for convergence
    #         if (np.abs(total_loss/n - prev_loss) < self.epsilon):
    #             break
    #         prev_loss = total_loss/n
            
    #     return avg_loss[0:i+1], error_rate[0:i+1]


    # def predict(self, test_loader, criterion):
    #     '''
    #     Function used to predict with the MLP

    #     test_loader: includes the feature matrix and classlabels corresponding to the test set,
    #     criterion: the loss function used.
    #     '''

    #     with torch.no_grad(): # no backprop step so turn off gradients
    #         correct = 0
    #         total_loss = 0
    #         for j,(images,labels) in enumerate(test_loader):

    #             # Compute prediction output and loss
    #             labels_pred = self.forward(images)
    #             labels_max = torch.argmax(labels_pred, axis=1)

    #             # Measure loss and error rate and record
    #             m = images.size()[0]
    #             loss = criterion(labels_pred, labels)
    #             correct += (labels_max == labels).sum().item()
    #             total_loss += loss.item() * m

    #     # Print/return test loss and error rate
    #     n = len(test_loader.dataset)
    #     avg_loss = total_loss/n
    #     error_rate = 1.0 - correct/n
    #     print(f"test loss: {avg_loss:>7f} test error rate: {error_rate:>7f}")
        
    #     return avg_loss, error_rate


