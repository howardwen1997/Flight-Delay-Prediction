import numpy as np

import torch
import torch.nn as nn

from matplotlib import pyplot as plt

from LSTM import LSTM

from Dataset_Loader import load_flight_seq

np.random.seed(2023)

is_cuda = torch.cuda.is_available()
print(is_cuda)

if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# load flight dataset
train_loader, test_loader = load_flight_seq(batch_size=20)

lstm_model = LSTM(input_size=798, hidden_size=120, batch_size = train_loader.batch_size, max_epochs=500)
lstm_model.fit(train_loader, nn.BCELoss(), optimizer=torch.optim.Adam(lstm_model.parameters(), lr=0.01))
lstm_model.predict(test_loader, nn.BCELoss())