import numpy as np
import bottleneck as bn
import torch

def calculate_ndcg(scores, ratings, k):
    # 将预测评分按降序排序，通过得到的序列对ratings继续排序，相当于得出了相关度rel-i
    _, sorted_indices = torch.sort(scores, descending=True)
    sorted_ratings = ratings[sorted_indices]

    # 计算 Discounted Cumulative Gain (DCG)
    # dcg = rel-i / log2(i+1)
    dcg = (sorted_ratings / torch.log2(torch.arange(2, len(sorted_ratings) + 2).float())).cumsum(dim=0)

    # 计算 Idealized Discounted Cumulative Gain (IDCG)
    sorted_ratings_ideal, _ = torch.sort(ratings, descending=True)
    idcg = (sorted_ratings_ideal / torch.log2(torch.arange(2, len(sorted_ratings_ideal) + 2).float())).cumsum(dim=0)

    # 计算 NDCG
    ndcg = dcg / idcg
    ndcg_k = ndcg[:k].mean()

    return ndcg_k.numpy()

def calculate_Recall_Preision_F1_OneCall(scores, ratings, k=5):
    # 获取Top-K位置的预测评分索引
    topk_indices = torch.topk(scores, k).indices

    # 将tensor转换为numpy数组
    topk_indices = topk_indices.cpu().numpy()
    ratings = ratings.cpu().numpy()
    scores = scores.cpu().numpy()

    # 计算预测评分和真实评分同时为True的数量
    true_positive = np.logical_and(np.abs(scores[topk_indices] - ratings[topk_indices]) <= 1, ratings[topk_indices] > 0).sum()

    # 计算真实评分为True的数量
    actual_positive = np.sum(ratings > 0)

    # 计算预测评分为True的数量
    predicted_positive = np.sum(scores > 0)

    # 计算召回率（Recall）
    # 预测的物品中用户需要的数量 / 用户需要的物品总数
    recall = true_positive / actual_positive if actual_positive > 0 else 0.0

    # 计算精确率（Precision）
    # 预测的物品中用户需要的数量 / 为用户预测的物品的总量
    precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0

    # 计算F1值
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 计算OneCall指标
    one_call = 1.0 if true_positive > 0 else 0.0

    return recall, precision, f1, one_call

# TODO: modify the auc calculation
def calculate_AUC_at_k(scores, ratings, k=5):
    # 获取排名前k个预测评分的索引
    topk_indices = torch.topk(scores, k).indices

    # 将预测评分和真实评分转换为 NumPy 数组
    scores_np = scores.cpu().numpy()
    ratings_np = ratings.cpu().numpy()

    # 构建二进制指示矩阵，表示在前k个位置上是否命中真实评分
    hit_matrix = np.zeros_like(scores_np, dtype=bool)
    hit_matrix[topk_indices] = True

    # 计算正样本和负样本的数量
    pos_samples = ratings_np > 0
    neg_samples = ratings_np == 0

    # 计算每个正样本的 AUC
    aucs = []
    for i, pos in enumerate(pos_samples):
        if pos:
            # 计算每个正样本的排名
            rank = np.sum(hit_matrix[:i + 1])

            # 计算每个正样本前面的负样本数量
            num_neg = np.sum(neg_samples[:i + 1])

            # 计算每个正样本的 AUC
            if num_neg != 0:
                auc = (rank - num_neg) / (k * num_neg)
            else:
                auc = 0.0
            aucs.append(auc)

    # 计算平均 AUC
    mean_auc = np.mean(aucs)

    return mean_auc

# 计算模型的大小
# TODO: 增加一个计算通讯量
def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    all_size = (param_size + buffer_size) / 1024 / 1024
    print('模型总大小为：{:.3f}MB'.format(all_size))
    return (param_size, param_sum, buffer_size, buffer_sum, all_size)

# 获得模型的计算量
# TODO: Flops 计算
def getModelCal(user_num, item_num, hidden_dim):
    # 根据计算矩阵分解模型的乘法操作数量来近似估计矩阵分解模型的计算量

    multiply_operation = (user_num * item_num * hidden_dim)

    print('模型近似计算量为：{:.3f}'.format(multiply_operation))

    return multiply_operation

if __name__ == '__main__':
    # 示例数据
    scores = torch.tensor([4.5, 2.3, 3.8, 1.9, 4.1])
    ratings = torch.tensor([5, 2, 4, 1, 3])

    # 计算 NDCG 值
    k = 3  # Top-K 位置
    ndcg = calculate_ndcg(scores, ratings, k)

    print("NDCG@{}: {:.4f}".format(k, ndcg))
