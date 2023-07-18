import torch
import os.path
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

FilePath = "data/ml-latest-small/ratings.csv"
FilePath2 = "data/ml-1m/ratings.dat"
# TODO: need to change to the larger dataset
FilePath3 = "data/ml-100k/u1.test"

class TrainDataset(Dataset):
    def __init__(self):
        super().__init__()
        # path = os.path.abspath(".")
        # movieLens-1m dataset
        # data = pd.read_csv(os.path.join(path, FilePath2), sep='::', engine='python', header=None,
        #                    names=['userId', 'movieId', 'rating', 'timestamp'])
        # movieLens-100k dataset
        # data = pd.read_csv(os.path.join(path, FilePath3), sep='\t', engine='python', header=None,
        #                    names=['userId', 'movieId', 'rating', 'timestamp'])

        # 读取CSV文件
        path = os.path.abspath(".")
        data = pd.read_csv(os.path.join(path, FilePath3), sep='\t', engine='python', header=None,
                         names=['userId', 'movieId', 'rating', 'timestamp'])

        # 获取用户数量和电影数量
        n_users = data['userId'].max()
        n_movies = data['movieId'].max()

        # 创建用户-电影评分矩阵
        ratings_matrix = np.zeros((n_users, n_movies))

        # 遍历数据框，填充评分矩阵
        for _, row in data.iterrows():
            user = row['userId'] - 1  # 将用户ID减1，使其从0开始
            movie = row['movieId'] - 1  # 将电影ID减1，使其从0开始
            ratings_matrix[user, movie] = 1

        self.train_dataset, _ = train_test_split(ratings_matrix, test_size=0.2, random_state=42)
        self.n = self.train_dataset.shape[0]
        self.m = self.train_dataset.shape[1]

    def __len__(self):
        return self.n * self.m

    # 自定义返回值，每次返回一个字典
    def __getitem__(self, index):
        user, item = index // self.m, index % self.m
        rating = self.train_dataset[user, item]

        return {'user': [user],
                'item': [item],
                'ratings': [rating]}

class TestDataset(Dataset):
    def __init__(self):
        super().__init__()
        # 读取CSV文件
        path = os.path.abspath(".")
        data = pd.read_csv(os.path.join(path, FilePath3), sep='\t', engine='python', header=None,
                           names=['userId', 'movieId', 'rating', 'timestamp'])

        # 获取用户数量和电影数量
        n_users = data['userId'].max()
        n_movies = data['movieId'].max()

        # 创建用户-电影评分矩阵
        ratings_matrix = np.zeros((n_users, n_movies))

        # 遍历数据框，填充评分矩阵
        for _, row in data.iterrows():
            user = row['userId'] - 1  # 将用户ID减1，使其从0开始
            movie = row['movieId'] - 1  # 将电影ID减1，使其从0开始
            ratings_matrix[user, movie] = 1

        _, self.test_dataset = train_test_split(ratings_matrix, test_size=0.2, random_state=42)
        self.n = self.test_dataset.shape[0]
        self.m = self.test_dataset.shape[1]

    def __len__(self):
        return self.n * self.m

    # 自定义返回值，每次返回一个字典
    def __getitem__(self, index):
        user, item = index // self.m, index % self.m
        rating = self.test_dataset[user, item]

        return {'user': [user],
                'item': [item],
                'ratings': [rating]}

class ClientsSampler(Dataset):
    def __init__(self, n):
        super().__init__()
        self.users_seq = np.arange(n)

    def __len__(self):
        return len(self.users_seq)

    def __getitem__(self, idx):
        return self.users_seq[idx]

# 测试对movieLens-100k的数据进行读取操作
def readFile():
    path = os.path.abspath(".")
    data = pd.read_csv(os.path.join(path, FilePath3), sep='\t', engine='python', header=None,
                       names=['userId', 'movieId', 'rating', 'timestamp'])
    print(data[:3])
    train_dataset, test_dataset = train_test_split(data, test_size=0.2)
    print(train_dataset.userId.max(), train_dataset.movieId.max())

def gptcode():
    # 读取CSV文件
    path = os.path.abspath(".")
    df = pd.read_csv(os.path.join(path, FilePath3), sep='\t', engine='python', header=None,
                       names=['userId', 'movieId', 'rating', 'timestamp'])

    # 获取用户数量和电影数量
    n_users = df['userId'].max()
    n_movies = df['movieId'].max()

    # 创建用户-电影评分矩阵
    ratings_matrix = np.zeros((n_users, n_movies))

    # 遍历数据框，填充评分矩阵
    for _, row in df.iterrows():
        user = row['userId'] - 1  # 将用户ID减1，使其从0开始
        movie = row['movieId'] - 1  # 将电影ID减1，使其从0开始
        ratings_matrix[user, movie] = 1

    # 打印评分矩阵
    # print(ratings_matrix)

    # 划分训练数据和测试数据
    train_data, test_data = train_test_split(ratings_matrix, test_size=0.2, random_state=42)

    # 打印训练数据和测试数据的形状
    print("训练数据形状:", train_data.shape)
    print("测试数据形状:", test_data.shape)


if __name__ == '__main__':
    # readFile()
    gptcode()