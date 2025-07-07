import numpy as np
import pandas as pd
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        path,
        split='train',
        train_size = 0.6,
        seq_len=24 * 4 * 4,
        label_len=24 * 4,
        pred_len=24 * 4,
        scale=True,
        inverse=False,
        random_state=42,
        is_timeencoded=True,
        frequency='d',
        features='MS',
        target='BP',
        random_sample = True,
        data_size=1.0
    ):
        assert split in ['train', 'val', 'test']
        self.path = path
        self.split = split
        self.train_size = train_size
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.scale = scale
        self.inverse = inverse
        self.random_state = random_state
        self.is_timeencoded = is_timeencoded
        self.frequency = frequency
        self.features = features
        self.target = target
        self.random_sample = random_sample
        self.data_size = data_size
        self.scaler = StandardScaler()
        self.prepare_data()

    def prepare_data(self):
        # Load and preprocess data
        df = pd.read_csv(self.path)
        df['Date'] = pd.to_datetime(df['Date'])

        # Split data into train, validation, and test sets
        indices = df.index.tolist()
        train_end = int(len(indices) * self.train_size)
        val_end = train_end + int(len(indices) * (1 - self.train_size) / 2)

        if self.split == 'train':
            split_indices = indices[:train_end]
            split_indices = split_indices[:int(len(split_indices) * self.data_size)]  # Apply data size
            
        elif self.split == 'val':
            split_indices = indices[train_end:val_end]
        else:  # 'test'
            split_indices = indices[val_end:]

        df_split = df.loc[split_indices]
        data_columns = df_split.columns[1:]  # Exclude the 'Date' column
        data = df_split[data_columns].values

        if self.scale:
            train_data = df.loc[indices[:train_end], data_columns].values
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        if self.inverse:
            data_y = df_split[data_columns].values
        else:
            data_y = data

        # Process time features
        timestamp = df_split[['Date']]
        if not self.is_timeencoded:
            timestamp['month'] = timestamp['Date'].dt.month
            timestamp['day'] = timestamp['Date'].dt.day
            timestamp['weekday'] = timestamp['Date'].dt.weekday
            if self.frequency in ['h', 't']:
                timestamp['hour'] = timestamp['Date'].dt.hour
            if self.frequency == 't':
                timestamp['minute'] = (timestamp['Date'].dt.minute // 15)
            timestamp_data = timestamp.drop(columns=['Date']).values
        else:
            timestamp_data = time_features(pd.to_datetime(timestamp['Date']), freq=self.frequency).transpose(1, 0)

        # Convert data to tensors
        self.time_series_x = torch.FloatTensor(data)
        self.time_series_y = torch.FloatTensor(data_y)
        self.timestamp = torch.FloatTensor(timestamp_data)

        total_length = len(self.time_series_x)
        self.indices = list(range(total_length - self.seq_len - self.pred_len + 1))

        if self.random_sample:
            np.random.seed(self.random_state)
            random.shuffle(self.indices)

        self.feature_names = data_columns

    def __getitem__(self, index):
        base_idx = self.indices[index]
        x_begin_index = base_idx
        x_end_index = x_begin_index + self.seq_len
        y_begin_index = x_end_index - self.label_len
        y_end_index = y_begin_index + self.label_len + self.pred_len

        x = self.time_series_x[x_begin_index:x_end_index]
        y = self.time_series_y[y_begin_index:y_end_index]
        x_timestamp = self.timestamp[x_begin_index:x_end_index]
        y_timestamp = self.timestamp[y_begin_index:y_end_index]

        return x, y, x_timestamp, y_timestamp

    def __len__(self):
        return len(self.time_series_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    @property
    def num_features(self):
        return self.time_series_x.shape[1]

    @property
    def columns(self):
        return self.feature_names

    
            
if __name__== "__main__":
    data_set = TimeSeriesDataset(
        path='data/main.csv'
    )
    data_loader = DataLoader(
        data_set,
        batch_size=32,
        shuffle=True,
        num_workers= 0,
        drop_last=True
    )
    for batch_idx, (x, y, x_timestamp, y_timestamp) in enumerate(data_loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        print(f"x_timestamp shape: {x_timestamp.shape}")
        print(f"y_timestamp shape: {y_timestamp.shape}")
        break

