import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler

# preprocess data to 2017-2019 only
"""
features = ['Year','Month','DayOfWeek','FlightDate','Reporting_Airline','Tail_Number'
            ,'Origin','Dest','DepTimeBlk','ArrTimeBlk','Cancelled','CancellationCode','Diverted'
            ,'CRSElapsedTime','Distance','ArrDel15']

data_2017_2019 = pd.DataFrame()

chunksize = 10**6
with pd.read_csv('airline.csv',
                    usecols=features,
                    encoding='unicode_escape',
                    chunksize=chunksize) as reader:
    for i, chunk in enumerate(reader):
        filter = (chunk['Year'] >= 2017) & (chunk['Year'] <= 2019)
        data_2017_2019 = pd.concat([data_2017_2019, chunk[filter]])
        #print(i/len(reader))
        print('row {}'.format(chunk.iloc[0].name))
        
data_2017_2019.to_csv('airline_2017_2019.csv', encoding='utf-8', index=False)   
"""

data_2017_2019 = pd.read_csv('airline_2017_2019.csv', encoding='utf-8', dtype={'FlightDate': str})    

# test if a flight sequence has at least one cancelled or diverted flights
data_2017_2019['cancel_max'] = data_2017_2019.groupby(['Tail_Number','FlightDate'])['Cancelled'].transform(np.max)
data_2017_2019['cancel_min'] = data_2017_2019.groupby(['Tail_Number','FlightDate'])['Cancelled'].transform(np.min)
#data_2017_2019['cancel_max'].sum()/data_2017_2019.shape[0]
#data_2017_2019[(data_2017_2019['cancel_max'] == 1) & (data_2017_2019['cancel_min'] == 0)].sort_values(by = ['Tail_Number','FlightDate','CRSDepTime'], inplace = False).head(20)
data_2017_2019['divert_max'] = data_2017_2019.groupby(['Tail_Number','FlightDate'])['Diverted'].transform(np.max)
data_2017_2019['divert_min'] = data_2017_2019.groupby(['Tail_Number','FlightDate'])['Diverted'].transform(np.min)
#data_2017_2019['cancel_max'].sum()/data_2019.shape[0]
#data_2017_2019[(data_2019['divert_max'] == 1) & (data_2017_2019['divert_min'] == 0)].sort_values(by = ['Tail_Number','FlightDate','CRSDepTime'], inplace = False).head(20)
data_2017_2019['null_target'] = data_2017_2019['ArrDel15'].isna().astype(int)
data_2017_2019['null_max'] = data_2017_2019.groupby(['Tail_Number','FlightDate'])['null_target'].transform(np.max)

data_2017_2019 = data_2017_2019[(data_2017_2019['cancel_max'] == 0) & (data_2017_2019['divert_max'] == 0) & (data_2017_2019['null_max'] == 0)]
data_2017_2019.drop(['Cancelled','CancellationCode','Diverted','cancel_max','cancel_min','divert_max','divert_min','null_target','null_max'], axis = 1, inplace = True)

data_2017_2019['flight_ct'] = data_2017_2019.groupby(['Tail_Number','FlightDate'])['Month'].transform('count')

# restricting a plane with 6 flights on a given day
data_2017_2019 = data_2017_2019[data_2017_2019['flight_ct'] == 6]
# generate groupid by date-tail number combination
#data_2017_2019['group_id'] = data_2017_2019.groupby(['FlightDate','Tail_Number']).ngroup()

data_2017_2019.sort_values(by = ['Tail_Number','FlightDate','DepTimeBlk'], inplace = True)

dummy_features = ['Month','DayOfWeek','Reporting_Airline','Origin','Dest','DepTimeBlk','ArrTimeBlk']
#print(data_2017_2019.head())
data_2017_2019 = pd.get_dummies(data_2017_2019, columns=dummy_features)
#print(data_2017_2019.head())
data_2017_2019 = data_2017_2019[0:600000]

# data_2017_2019=(data_2017_2019-data_2017_2019.mean())/data_2017_2019.std()
# data_2017_2019.head()

# number of obs
n_obs = len(data_2017_2019[['Tail_Number','FlightDate']].value_counts())
data_2017_2019.drop(['Year','FlightDate','Tail_Number','flight_ct'], axis = 1, inplace = True)
n_features = len(data_2017_2019.columns) - 1
#print(data_2017_2019.columns)

target_column = data_2017_2019['ArrDel15'].copy()
col_names = data_2017_2019.columns
scaler = StandardScaler()
scaler.fit(data_2017_2019.to_numpy())
data_2017_2019 = scaler.transform(data_2017_2019.to_numpy())
data_2017_2019 = pd.DataFrame(data_2017_2019, columns = col_names)
data_2017_2019['ArrDel15'] = target_column.values

# empty features and target placeholder
sequence_data_features = np.zeros(shape=(n_obs, 6, n_features))
sequence_data_target = np.zeros(shape=(n_obs, 6, 1))

for index, num in enumerate(range(0, len(data_2017_2019)-12, 6)):
    sequence_data_features[index] = data_2017_2019[num: num+6].drop(['ArrDel15'], axis = 1)
    sequence_data_target[index] = data_2017_2019[num: num+6]['ArrDel15'].values.reshape(6,1)

print(sequence_data_features.shape)
print(sequence_data_target.shape)
print(data_2017_2019['ArrDel15'].value_counts()/data_2017_2019.shape[0])

class myDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.x = torch.tensor(self.x, dtype = torch.float32)
        self.y = torch.tensor(self.y, dtype = torch.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
#data = myDataset(sequence_data_features, sequence_data_target)
#print(data[0])

def load_flight_seq(batch_size=64):
    data = myDataset(sequence_data_features, sequence_data_target)

    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    trainset, testset = random_split(data, [train_size, test_size])

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader