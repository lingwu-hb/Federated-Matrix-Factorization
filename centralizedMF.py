# 集中式矩阵分解算法
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import TrainDataset, TestDataset
from model import MFModel
from metric import *

class centralizedMF:
    def __init__(self, args):
        self.trainData = TrainDataset()
        self.testData = TestDataset()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.n = self.trainData.n
        self.m = self.trainData.m
        self.test_data = DataLoader(
            self.testData,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False)
        self.model = MFModel(self.n, self.m, args.hiddenDim)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            all_loss = 0
            dataBatch = DataLoader(self.trainData, batch_size=self.batch_size, shuffle=True)
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

        #     # evaluate phase
        #     ndcg5_list = []
        #     recall5_list = []
        #     precision5_list = []
        #     f1_list = []
        #     oneCall_list = []
        #     auc_list = []
        #
        #     self.model.eval()
        #     with torch.no_grad():
        #         for batch in self.test_data:
        #             batch_dict = dict([(k, v[0].float().to(self.device)) for k, v in batch.items()])
        #             users = batch_dict['user'].int()
        #             items = batch_dict['item'].int()
        #             ratings = batch_dict['ratings']
        #
        #             scores = self.model({'user': users, 'item': items})
        #             scores[users, items] = -np.inf
        #             recon_batch = scores.cpu().numpy()
        #             ratings = ratings.cpu().numpy()
        #
        #             n_5 = NDCG_binary_at_k_batch(recon_batch, ratings, 5)
        #             r_5, p_5, f_5, o_5 = Recall_Precision_F1_OneCall_at_k_batch(recon_batch, ratings, 5)
        #             auc_b = AUC_at_k_batch(users.cpu().numpy(), recon_batch, ratings)
        #
        #             ndcg5_list.append(n_5)
        #             recall5_list.append(r_5)
        #             precision5_list.append(p_5)
        #             f1_list.append(f_5)
        #             oneCall_list.append(o_5)
        #             auc_list.append(auc_b)
        #
        #     ndcg5_list = np.concatenate(ndcg5_list)
        #     recall5_list = np.concatenate(recall5_list)
        #     precision5_list = np.concatenate(precision5_list)
        #     f1_list = np.concatenate(f1_list)
        #     oneCall_list = np.concatenate(oneCall_list)
        #     auc_list = np.concatenate(auc_list)
        #
        #     ndcg5_list[np.isnan(ndcg5_list)] = 0
        #     ndcg5 = np.mean(ndcg5_list)
        #     recall5_list[np.isnan(recall5_list)] = 0
        #     recall5 = np.mean(recall5_list)
        #     precision5_list[np.isnan(precision5_list)] = 0
        #     precision5 = np.mean(precision5_list)
        #     f1_list[np.isnan(f1_list)] = 0
        #     f1 = np.mean(f1_list)
        #     oneCall_list[np.isnan(oneCall_list)] = 0
        #     oneCAll = np.mean(oneCall_list)
        #     auc_list[np.isnan(auc_list)] = 0
        #     auc = np.mean(auc_list)
        #
        #     print(
        #         "Epoch: {:3d} | Pre@5: {:5.4f} | Rec@5: {:5.4f} | F1@5: {:5.4f} | NDCG@5: {:5.4f} | 1-call@5: {:5.4f} | AUC: {:5.4f}".format(
        #             epoch + 1, precision5, recall5, f1, ndcg5, oneCAll, auc), flush=True)
        #
        #     if ndcg5 > best_ndcg:
        #         best_ndcg = ndcg5
        #         best_epoch = epoch + 1
        #         patience = self.early_stop
        #     else:
        #         patience -= 1
        #         if patience == 0:
        #             break
        #
        # print('epoch of best ndcg@5({:5.4f})'.format(best_ndcg), best_epoch, flush=True)

        # calculate the calculated amount and size of the model
        getModelSize(self.model)
        # getModelCal(self.model, self.batch_size, self.m, self.n)




