import os.path
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

FilePath = "data/ml-latest-small/ratings.csv"
FilePath2 = "data/ml-1m/ratings.dat"
FilePath3 = "data/ml-100k/u1.test"

class TrainDataset(Dataset):
    def __init__(self):
        super().__init__()
        path = os.path.abspath(".")
        # movieLens-1m dataset
        # data = pd.read_csv(os.path.join(path, FilePath2), sep='::', engine='python', header=None,
        #                    names=['userId', 'movieId', 'rating', 'timestamp'])
        # movieLens-100k dataset
        # data = pd.read_csv(os.path.join(path, FilePath3), sep='\t', engine='python', header=None,
        #                    names=['userId', 'movieId', 'rating', 'timestamp'])

        # TODO: need to modify
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

        self.train_dataset, _ = train_test_split(data, test_size=0.2)
        self.n = self.train_dataset.userId.max()
        self.m = self.train_dataset.movieId.max()
        self.train_dataset['userId'] = self.train_dataset['userId']-1
        self.train_dataset['movieId'] = self.train_dataset['movieId']-1

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

class TestDataset(Dataset):
    def __init__(self):
        super().__init__()
        path = os.path.abspath(".")
        data = pd.read_csv(os.path.join(path, FilePath3), sep='\t', engine='python', header=None,
                           names=['userId', 'movieId', 'rating', 'timestamp'])
        _, self.test_dataset = train_test_split(data, test_size=0.2)
        self.n = self.test_dataset.userId.max()
        self.m = self.test_dataset.movieId.max()
        self.test_dataset['userId'] = self.test_dataset['userId']-1
        self.test_dataset['movieId'] = self.test_dataset['movieId']-1

    def __len__(self):
        return len(self.test_dataset)

    # 自定义返回值，每次返回一个字典
    def __getitem__(self, index):
        users = [self.test_dataset.iloc[index].userId]
        items = [self.test_dataset.iloc[index].movieId]
        ratings = [self.test_dataset.iloc[index].rating]

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
    print(ratings_matrix)


if __name__ == '__main__':
    # readFile()
    gptcode()