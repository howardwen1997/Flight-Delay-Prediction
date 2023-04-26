import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from LSTM_Class import LSTM
from LSTM_Dataset_Loader import load_flight_seq
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
np.random.seed(2023)

# load flight dataset
train_loader, val_loader, test_loader = load_flight_seq(batch_size=64)
input_size = train_loader.dataset[0][0].shape[1]
max_epochs =500

# early stopping
patience_time = 5

# metrics placeholder
metrics_dict = {}

for num_lstm_layer in [1,2]:
    for hidden_size in [30,50]:

        lstm_model = LSTM(input_size=input_size, hidden_size=50, num_layers=num_lstm_layer)
        print(f'Number of LSTM layers: {num_lstm_layer}, Hidden size: {hidden_size}')
        for epoch in range(1, max_epochs+1):
            print(f'\nEpoch {epoch}')
            lstm_model.fit(train_loader, nn.BCEWithLogitsLoss(pos_weight = torch.tensor(4.618)), optimizer=torch.optim.Adam(lstm_model.parameters(), lr=0.001))
            lstm_model.predict(val_loader, nn.BCEWithLogitsLoss(pos_weight = torch.tensor(4.618)), set='validation')
            if epoch > patience_time:
                if lstm_model.val_loss_epoch[-1] > lstm_model.val_loss_epoch[-3]:
                    print('early stopping!')
                    break

        metrics_dict[(num_lstm_layer, hidden_size)] = lstm_model.predict(test_loader, nn.BCEWithLogitsLoss(pos_weight = torch.tensor(4.618)), set='test')                
        
print(metrics_dict)        
