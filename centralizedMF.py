# 集中式矩阵分解算法
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset
from model import MFModel
from metric import *
from evaluate import evaluateModel

class centralizedMF:
    def __init__(self, args):
        self.trainData = TrainDataset()
        self.testData = TestDataset()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.early_stop = args.early_stop
        self.n = self.trainData.n
        self.m = self.trainData.m
        self.hiddenDim = args.hiddenDim
        self.test_data = DataLoader(
            self.testData,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
            drop_last=True)
        self.model = MFModel(self.n, self.m, self.hiddenDim)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    def train(self):
        best_ndcg = -np.inf
        best_epoch = 0
        patience = self.early_stop
        for epoch in range(self.epochs):
            self.model.train()
            all_loss = 0
            dataBatch = DataLoader(self.trainData,
                                   batch_size=self.batch_size,
                                   shuffle=True,
                                   drop_last=True)
            for batch in dataBatch:
                # 直接调用model函数，会调用模型的forward函数，返回对结果的预测值
                scores = self.model(dict([(k, v[0].float().to(self.device)) for k, v in batch.items()]))
                # 梯度清零
                self.optimizer.zero_grad()
                # 计算损失值
                loss = self.model.loss_function(scores, batch['ratings'][0].float().to(self.device))
                # 反向传播，计算梯度
                loss.backward()
                # 调用优化函数优化整个模型
                self.optimizer.step()
                all_loss += loss.item()
            print('epoch%d - loss%f' % (epoch, all_loss / len(self.trainData)))

            # eva-function(self.model, self.test_data) -> ndcg5_mean
            ndcg5_mean = evaluateModel(self.model, epoch, self.test_data, self.device)

            if ndcg5_mean > best_ndcg:
                best_ndcg = ndcg5_mean
                best_epoch = epoch + 1
                patience = self.early_stop
            else:
                patience -= 1
                if patience == 0:
                    break
        print('epoch of best ndcg@5({:5.4f})'.format(best_ndcg), best_epoch, flush=True)

        # 打印出预测值和真实值进行比较
        prection = []
        real_score = []
        with torch.no_grad():
            for batch in self.test_data:
                batch_dict = dict([(k, v[0].float().to(self.device)) for k, v in batch.items()])
                users = batch_dict['user'].int()
                items = batch_dict['item'].int()
                ratings = batch_dict['ratings']
                scores = self.model({'user': users, 'item': items})
                prection.append(scores.tolist())
                real_score.append(ratings.tolist())
            print("预测值为:", np.array(prection))
            print("真实值为:", np.array(real_score))

        # calculate the calculated amount and size of the model
        getModelSize(self.model)
        getModelCal(self.m, self.n, self.hiddenDim)




