import numpy as np

import torch
import torch.nn as nn

class FlightMLP(nn.Module):
    
    def __init__(self, layer_sizes, dropout=False):
        '''
        layer_sizes (array): Array of hidden layer sizes
        dropout (optional, default=False): Whether to use dropout layers
        '''
        
        super(FlightMLP, self).__init__()
        
        input_size = 31
        
        # make layers
        input = [nn.Linear(input_size, layer_sizes[0]), nn.ReLU()]
        for i in range(len(layer_sizes)-1):
            input.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            input.append(nn.ReLU())
            if dropout:
                input.append(nn.Dropout(p=0.1))
                
        input.append(nn.Linear(layer_sizes[-1], 1))

        self.mlp = nn.Sequential(*input)


    def forward(self, x):
        return self.mlp(x)
