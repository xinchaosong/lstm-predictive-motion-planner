import torch
from torch.utils.data import Dataset
import pandas as pd


class CsvDataset(Dataset):
    def __init__(self, data_file):
        super(CsvDataset, self).__init__()

        print("Loading the data...")

        df_trajectories = pd.read_csv(data_file, header=None)
        self.data = torch.tensor(df_trajectories.values, dtype=torch.float).unsqueeze(2)

        print("Data loaded successfully.")
        print("Data set dimension:", self.data.shape)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]
