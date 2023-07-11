import torch
from torch.utils.data import DataLoader
import numpy as np
from model import MFModel
from dataset import TrainDataset, TestDataset, ClientsSampler
from metric import Recall_Precision_F1_OneCall_at_k_batch, NDCG_binary_at_k_batch, AUC_at_k_batch


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
            clients_data_dict = dict([(k, torch.from_numpy(np.array(v)).float().to(self.device))
                                      for k, v in self.trainData[uid].items()])
            self.optimizer.zero_grad()
            scores = self.model(clients_data_dict)
            # 需要对item进行更新
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
        self.early_stop = args.early_stop
        self.test_data = DataLoader(
            self.clients.testData,
            batch_size=self.batch_size,
            num_workers=2,
            pin_memory=True,
            shuffle=False)
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
        best_ndcg = -np.inf
        best_epoch = 0
        patience = self.early_stop
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
            ndcg5_list = []
            recall5_list = []
            precision5_list = []
            f1_list = []
            oneCall_list = []
            auc_list = []

            self.model.eval()
            with torch.no_grad():
                for batch in self.test_data:
                    batch_dict = dict([(k, torch.from_numpy(np.array(v)).float().to(self.device))
                                  for k, v in batch.items()])
                    users = batch_dict['user'].to(self.device)
                    items = batch_dict['item'].to(self.device)
                    ratings = batch_dict['ratings'].to(self.device)

                    scores = self.model({'user': users, 'item': items})
                    scores[users, items] = -np.inf
                    recon_batch = scores.cpu().numpy()
                    ratings = ratings.cpu().numpy()

                    n_5 = NDCG_binary_at_k_batch(recon_batch, ratings, 5)
                    r_5, p_5, f_5, o_5 = Recall_Precision_F1_OneCall_at_k_batch(recon_batch, ratings, 5)
                    auc_b = AUC_at_k_batch(users.cpu().numpy(), recon_batch, ratings)

                    ndcg5_list.append(n_5)
                    recall5_list.append(r_5)
                    precision5_list.append(p_5)
                    f1_list.append(f_5)
                    oneCall_list.append(o_5)
                    auc_list.append(auc_b)

            ndcg5_list = np.concatenate(ndcg5_list)
            recall5_list = np.concatenate(recall5_list)
            precision5_list = np.concatenate(precision5_list)
            f1_list = np.concatenate(f1_list)
            oneCall_list = np.concatenate(oneCall_list)
            auc_list = np.concatenate(auc_list)

            ndcg5_list[np.isnan(ndcg5_list)] = 0
            ndcg5 = np.mean(ndcg5_list)
            recall5_list[np.isnan(recall5_list)] = 0
            recall5 = np.mean(recall5_list)
            precision5_list[np.isnan(precision5_list)] = 0
            precision5 = np.mean(precision5_list)
            f1_list[np.isnan(f1_list)] = 0
            f1 = np.mean(f1_list)
            oneCall_list[np.isnan(oneCall_list)] = 0
            oneCAll = np.mean(oneCall_list)
            auc_list[np.isnan(auc_list)] = 0
            auc = np.mean(auc_list)

            print(
                "Epoch: {:3d} | Pre@5: {:5.4f} | Rec@5: {:5.4f} | F1@5: {:5.4f} | NDCG@5: {:5.4f} | 1-call@5: {:5.4f} | AUC: {:5.4f}".format(
                    epoch + 1, precision5, recall5, f1, ndcg5, oneCAll, auc), flush=True)

            if ndcg5 > best_ndcg:
                best_ndcg = ndcg5
                best_epoch = epoch + 1
                patience = self.early_stop
            else:
                patience -= 1
                if patience == 0:
                    break
        print('epoch of best ndcg@5({:5.4f})'.format(best_ndcg), best_epoch, flush=True)


