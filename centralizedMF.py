# 集中式矩阵分解算法
import torch
from torch.utils.data import DataLoader
from dataset import ClientsDataset
from model import MFModel

class centralizedMF:
    def __init__(self, args):
        self.data = ClientsDataset()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.n = self.data.n
        self.m = self.data.m
        self.model = MFModel(self.n, self.n, args.hiddenDim)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            all_loss = 0
            dataBatch = DataLoader(self.data.train_dataset, batch_size=self.batch_size, shuffle=True)
            for batch in dataBatch:
                # 直接调用model函数，会调用模型的forward函数，返回对结果的预测值
                scores = self.model(dict([(k, v.to(self.device)) if torch.is_tensor(v) else (k, v) for k, v in batch.items()]))
                # 梯度清零
                self.optimizer.zero_grad()
                # 计算损失值
                loss = self.model.loss_function(scores, batch['label'].to(self.device))
                # 反向传播，计算梯度
                loss.backward()
                # 调用优化函数优化整个模型
                self.optimizer.step()
                all_loss += loss.item()
            print('epoch%d - loss%f' % (epoch, all_loss / len(self.data.train_dataset)))

        self.model.eval()


