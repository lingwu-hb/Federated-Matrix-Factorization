import os.path
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import Dataset

FilePath = "data/ml-latest-small/ratings.csv"

class ClientsDataset(Dataset):
    def __init__(self):
        super().__init__()
        path = os.path.abspath(".")
        data = pd.read_csv(os.path.join(path, FilePath))
        self.train_dataset, self.test_dataset = train_test_split(data, test_size=0.2)
        self.n = self.train_dataset.userId.max()
        self.m = self.train_dataset.movieId.max()
        data['userId'] = data['userId']-1
        data['movieId'] = data['movieId']-1

    def __len__(self):
        return len(self.train_dataset)

    # 自定义返回值，每次返回一个字典
    def __getitem__(self, index):
        users = [self.train_dataset.iloc[index].userId]
        items = [self.train_dataset.iloc[index].movieId]
        ratings = [self.train_dataset.iloc[index].rating]

        return {'user': users,
                'item': items,
                'ratings': ratings}

class ClientsSampler(Dataset):
    def __init__(self, n):
        super().__init__()
        self.users_seq = np.arange(n)

    def __len__(self):
        return len(self.users_seq)

    def __getitem__(self, idx):
        return self.users_seq[idx]
