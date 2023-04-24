import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.model_selection import train_test_split

# preprocess data to 2017-2019 only
"""
features = ['Year','Month','DayOfWeek','FlightDate','Reporting_Airline','Tail_Number'
            ,'Origin','Dest','DepTimeBlk','ArrTimeBlk','Cancelled','CancellationCode','Diverted'
            ,'CRSElapsedTime','Distance','ArrDel15']

df = pd.DataFrame()

chunksize = 10**6
with pd.read_csv('airline.csv',
                    usecols=features,
                    encoding='unicode_escape',
                    chunksize=chunksize) as reader:
    for i, chunk in enumerate(reader):
        filter = (chunk['Year'] >= 2017) & (chunk['Year'] <= 2019)
        df = pd.concat([df, chunk[filter]])
        #print(i/len(reader))
        print('row {}'.format(chunk.iloc[0].name))
        
df.to_csv('airline_2017_2019.csv', encoding='utf-8', index=False)   
"""

df = pd.read_csv('airline_2017_2019.csv', encoding='utf-8', dtype={'FlightDate': str})    

df.loc[~df['Origin'].isin(list(df['Origin'].value_counts()[:20].index)),'Origin'] = 'AllOthers'
df.loc[~df['Dest'].isin(list(df['Dest'].value_counts()[:20].index)),'Dest'] = 'AllOthers'

# test if a flight sequence has at least one cancelled or diverted flights
df['cancel_max'] = df.groupby(['Tail_Number','FlightDate'])['Cancelled'].transform(np.max)
df['cancel_min'] = df.groupby(['Tail_Number','FlightDate'])['Cancelled'].transform(np.min)
df['divert_max'] = df.groupby(['Tail_Number','FlightDate'])['Diverted'].transform(np.max)
df['divert_min'] = df.groupby(['Tail_Number','FlightDate'])['Diverted'].transform(np.min)
df['null_target'] = df['ArrDel15'].isna().astype(int)
df['null_max'] = df.groupby(['Tail_Number','FlightDate'])['null_target'].transform(np.max)

df = df[(df['cancel_max'] == 0) & (df['divert_max'] == 0) & (df['null_max'] == 0)]
df.drop(['Cancelled','CancellationCode','Diverted','cancel_max','cancel_min','divert_max','divert_min','null_target','null_max'], axis = 1, inplace = True)
df['flight_ct'] = df.groupby(['Tail_Number','FlightDate'])['Month'].transform('count')

# restricting a plane with 6 flights on a given day
df = df[df['flight_ct'] == 6]
#generate group ids to be used for train test split later
df['group_id'] = df.groupby(['FlightDate','Tail_Number']).ngroup()

# train, val, test split
test_ratio = 0.1
test_size = int(np.floor(np.max(df['group_id']) * test_ratio))

index_list = np.random.choice(df['group_id'].values, size=(2,test_size), replace=False)
test_index = index_list[0]
val_index = index_list[1]

df_train = df[~df['group_id'].isin(np.concatenate((test_index, val_index), axis=None))]
df_val = df[df['group_id'].isin(val_index)]
df_test = df[df['group_id'].isin(test_index)]

# DepTimeBlk and ArrTimeBlk  target encoding
df_train['DepTimeBlk_target_encoding'] = df_train.groupby(['DepTimeBlk'])[['ArrDel15']].transform('mean')
deptime_blk = df_train.groupby(['DepTimeBlk'])[['ArrDel15']].mean().rename(columns={'ArrDel15':'DepTimeBlk_target_encoding'})
df_val = df_val.merge(deptime_blk, on = ['DepTimeBlk'])
df_test = df_test.merge(deptime_blk, on = ['DepTimeBlk'])

df_train['ArrTimeBlk_target_encoding'] = df_train.groupby(['ArrTimeBlk'])[['ArrDel15']].transform('mean')
deptime_blk = df_train.groupby(['ArrTimeBlk'])[['ArrDel15']].mean().rename(columns={'ArrDel15':'ArrTimeBlk_target_encoding'})

df_val = df_val.merge(deptime_blk, on = ['ArrTimeBlk'])
df_test = df_test.merge(deptime_blk, on = ['ArrTimeBlk'])

for data in [df_train, df_val, df_test]:
    data.sort_values(by = ['Tail_Number','FlightDate','DepTimeBlk'], inplace = True)
    data.drop(['Year','FlightDate','Tail_Number','DepTimeBlk', 'ArrTimeBlk','flight_ct','group_id'], axis=1, inplace=True)

dummy_features = ['Month','DayOfWeek','Reporting_Airline','Origin','Dest']

df_train = pd.get_dummies(df_train, columns=dummy_features, dtype=float)
df_val = pd.get_dummies(df_val, columns=dummy_features, dtype=float)
df_test = pd.get_dummies(df_test, columns=dummy_features, dtype=float)

# standardize features
target_column_train = df_train['ArrDel15'].copy()
target_column_val = df_val['ArrDel15'].copy()
target_column_test = df_test['ArrDel15'].copy()

col_names = df_train.columns
scaler = StandardScaler()
scaler.fit(df_train.to_numpy())

df_train = scaler.transform(df_train.to_numpy())
df_train = pd.DataFrame(df_train, columns = col_names)
df_train['ArrDel15'] = target_column_train.values

df_val = scaler.transform(df_val.to_numpy())
df_val = pd.DataFrame(df_val, columns = col_names)
df_val['ArrDel15'] = target_column_val.values

df_test = scaler.transform(df_test.to_numpy())
df_test = pd.DataFrame(df_test, columns = col_names)
df_test['ArrDel15'] = target_column_test.values

# convert data into time sequence format
n_obs_train = int(df_train.shape[0]/6)
n_features = len(df_train.columns) - 1

sequence_train_features = np.zeros(shape=(n_obs_train, 6, n_features))
sequence_train_target = np.zeros(shape=(n_obs_train, 6, 1))

for index, num in enumerate(range(0, len(df_train)-6, 6)):
    sequence_train_features[index] = df_train[num: num+6].drop(['ArrDel15'], axis = 1)
    sequence_train_target[index] = df_train[num: num+6]['ArrDel15'].values.reshape(6,1)
    
n_obs_val = int(df_val.shape[0]/6)

sequence_val_features = np.zeros(shape=(n_obs_val, 6, n_features))
sequence_val_target = np.zeros(shape=(n_obs_val, 6, 1))

for index, num in enumerate(range(0, len(df_val)-6, 6)):
    sequence_val_features[index] = df_val[num: num+6].drop(['ArrDel15'], axis = 1)
    sequence_val_target[index] = df_val[num: num+6]['ArrDel15'].values.reshape(6,1)    
    
n_obs_test = int(df_test.shape[0]/6)

sequence_test_features = np.zeros(shape=(n_obs_test, 6, n_features))
sequence_test_target = np.zeros(shape=(n_obs_test, 6, 1))

for index, num in enumerate(range(0, len(df_test)-6, 6)):
    sequence_test_features[index] = df_test[num: num+6].drop(['ArrDel15'], axis = 1)
    sequence_test_target[index] = df_test[num: num+6]['ArrDel15'].values.reshape(6,1)   

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

def load_flight_seq(batch_size):
    trainset = myDataset(sequence_train_features, sequence_train_target)
    valset = myDataset(sequence_val_features, sequence_val_target)
    testset = myDataset(sequence_test_features, sequence_test_target)

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader