import numpy as np

import torch
import torch.nn as nn

# Fully connected neural network with one hidden layer
class FlightMLP(nn.Module):
    
    def __init__(self):
        '''
        max_epochs: maximum number of epochs to run gradient descent
        '''
        
        super(FlightMLP, self).__init__()
        
        # sizes (arbitrar)
        input_size = 31
        hl1 = 100
        hl2 = 50

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hl1),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(hl1, hl2),
            # nn.ReLU(),
            # nn.Dropout(),
            # nn.Linear(hl1, hl2),
            # nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(hl1, 1),
        )


    def forward(self, x):
        return self.mlp(x)


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

