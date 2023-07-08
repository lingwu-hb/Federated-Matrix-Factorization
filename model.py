from torch import nn
import math

class MFModel(nn.Module):
    def __init__(self, user_num, item_num, hidden_dim) -> None:
        super().__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dim = hidden_dim

        self.user_emb = nn.Embedding(user_num, self.dim) # 用户数量乘以隐藏维度的矩阵
        self.item_emb = nn.Embedding(item_num, self.dim)

        # 均方误差函数
        self.loss_function = nn.MSELoss()
        # 参数初始化
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化参数，使模型从一个较好的起始点开始训练
        stdv = 1 / math.sqrt(self.dim)
        for weight in self.parameters():
            nn.init.normal_(weight.data, 0, stdv)

    def forward(self, data):
        data['user'] = data['user'].int()
        data['item'] = data['item'].int()
        user_emb = self.user_emb(data['user'])
        item_emb = self.item_emb(data['item'])

        # 前向传播，将用户和物品的向量进行内积作为预测值
        # sum(-1)表示将矩阵按行进行求和操作
        return (user_emb * item_emb).sum(-1)