import torch
from torch.utils.data import DataLoader
import numpy as np
from model import MFModel
from dataset import TrainDataset, TestDataset, ClientsSampler
from metric import *
from evaluate import evaluateModel

class Clients:
    def __init__(self, args):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.trainData = TrainDataset()
        self.testData = TestDataset()
        self.n = self.trainData.n
        self.m = self.trainData.m
        self.model = MFModel(self.n, self.m, args.hiddenDim)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    # local train for model
    def train(self, uids, model_param_state_dict):
        # receive model parameters from the server
        self.model.load_state_dict(model_param_state_dict)
        # each client computes gradients using its private data
        clients_grads = {}
        all_loss = 0
        for uid in uids:
            # 把 tensor 类型换成列表
            # list -> numpy -> tensor(torch.from_numpy) -> to(self.device)
            uid = uid.tolist()
            clients_data_dict = dict([(k, torch.from_numpy(np.array(v)).float().to(self.device))
                                      for k, v in self.trainData[uid].items()])
            self.optimizer.zero_grad()
            scores = self.model(clients_data_dict)
            # 需要对item进行更新
            loss = self.model.loss_function(scores, clients_data_dict['ratings'])
            loss.backward()
            self.optimizer.step()
            all_loss += loss.item()
            grad_u = {}
            for name, param in self.model.named_parameters():
                grad_u[name] = param.grad.detach().clone()
            clients_grads[uid] = grad_u
        return clients_grads, all_loss

class Server:
    def __init__(self, args, clients):
        self.clients = clients
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.early_stop = args.early_stop
        self.m = self.clients.m
        self.n = self.clients.n
        self.hiddenDim = args.hiddenDim
        self.test_data = DataLoader(
            self.clients.testData,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False)
        self.model = MFModel(self.n, self.m, self.hiddenDim)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    # 服务器端对参数进行聚合
    def aggregate_gradients(self, clients_grads):
        clients_num = len(clients_grads)
        aggregated_gradients = {}
        for uid, grads_dict in clients_grads.items():
            for name, grad in grads_dict.items():
                if name in aggregated_gradients:
                    aggregated_gradients[name] = aggregated_gradients[name] + grad / clients_num
                else:
                    aggregated_gradients[name] = grad / clients_num

        for name, param in self.model.named_parameters():
            if param.grad is None:
                param.grad = aggregated_gradients[name].detach().clone()
            else:
                param.grad += aggregated_gradients[name]

    def train(self):
        best_ndcg = -np.inf
        best_epoch = 0
        patience = self.early_stop
        for epoch in range(self.epochs):
            # train phase
            self.model.train()
            all_loss = 0
            uid_seq = DataLoader(ClientsSampler(self.n), batch_size=self.batch_size, shuffle=True)
            for uids in uid_seq:
                # sample clients to train the model
                self.optimizer.zero_grad()
                # send the model to the clients and let them start training
                clients_grads, loss = self.clients.train(uids, self.model.state_dict())
                # aggregate the received gradients
                self.aggregate_gradients(clients_grads)
                # update the model
                self.optimizer.step()
                all_loss += loss
            print('epoch%d - loss%f' % (epoch, all_loss / self.n))

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