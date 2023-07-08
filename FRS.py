import torch
from torch.utils.data import DataLoader
import numpy as np
from model import MFModel
from dataset import TrainDataset, TestDataset, ClientsSampler

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
        for uid in uids:
            # 把 tensor 类型换成列表
            # list -> numpy -> tensor(torch.from_numpy) -> to(self.device)
            uid = uid.tolist()
            clients_data_dict = dict([(k, torch.from_numpy(np.array(v)).float().to(self.device)) for k, v in self.trainData[uid].items()])
            scores = self.model(clients_data_dict)
            loss = self.model.loss_function(scores, clients_data_dict['ratings'])
            loss.backward()
            self.optimizer.step()
            grad_u = {}
            for name, param in self.model.named_parameters():
                grad_u[name] = param.grad.detach().clone()
            clients_grads[uid] = grad_u
        return clients_grads

class Server:
    def __init__(self, args, clients):
        self.clients = clients
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.model = MFModel(self.clients.n, self.clients.m, args.hiddenDim)
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
        for epoch in range(self.epochs):
            # train phase
            self.model.train()
            uid_seq = DataLoader(ClientsSampler(self.clients.n), batch_size=self.batch_size, shuffle=True)
            for uids in uid_seq:
                # sample clients to train the model
                self.optimizer.zero_grad()
                # send the model to the clients and let them start training
                clients_grads = self.clients.train(uids, self.model.state_dict())
                # aggregate the received gradients
                self.aggregate_gradients(clients_grads)
                # update the model
                self.optimizer.step()

            # evaluate phase
            self.model.eval()
            # perform evaluation on the test set
            # HR / Recall / Precision and so on

