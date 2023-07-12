import numpy as np
import bottleneck as bn
from torchsummary import summary
import torch

def calculate_ndcg(scores, ratings, k):
    # 将预测评分和真实评分按降序排序
    _, sorted_indices = torch.sort(scores, descending=True)
    sorted_ratings = ratings[sorted_indices]

    # 计算 Discounted Cumulative Gain (DCG)
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
    true_positive = np.logical_and(scores[topk_indices] > 0, ratings[topk_indices] > 0).sum()

    # 计算真实评分为True的数量
    actual_positive = np.sum(ratings > 0)

    # 计算预测评分为True的数量
    predicted_positive = np.sum(scores > 0)

    # 计算召回率（Recall）
    recall = true_positive / actual_positive if actual_positive > 0 else 0.0

    # 计算精确率（Precision）
    precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0

    # 计算F1值
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # 计算OneCall指标
    one_call = 1.0 if true_positive > 0 else 0.0

    return recall, precision, f1, one_call


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
            auc = (rank - num_neg) / (k * num_neg)
            aucs.append(auc)

    # 计算平均 AUC
    mean_auc = np.mean(aucs)

    return mean_auc

def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=5):
    '''
    normalized discounted cumulative gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    # 获取预测评分矩阵的行数，即批次中的用户数量。
    batch_users = X_pred.shape[0]
    # 对预测评分矩阵按照分数进行分区，保留每行的前k个最高分数的索引。
    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    # 根据上一步得到的索引，从预测评分矩阵中取出对应的前k个最高分数的部分。
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :k]]
    # 对前一步得到的部分进行降序排列，得到每个用户在前k个推荐中的索引。
    idx_part = np.argsort(-topk_part, axis=1)
    # topk predicted score
    # 根据上一步得到的索引，从完整的索引中获取用户在完整预测评分矩阵中的前k个最高分数的索引。
    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
    # build the discount template
    # 计算折扣系数，即每个位置的折扣系数。
    tp = 1. / np.log2(np.arange(2, k + 2))
    # 计算折现累积增益（Discounted Cumulative Gain，DCG）指标，将每个用户的真实评分与对应位置的折扣系数相乘，并求和。
    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis], idx_topk] * tp).sum(axis=1)
    # 返回标准化的折现累积增益（NDCG）指标，将DCG除以IDCG，以便进行标准化。该指标越接近1，表示推荐结果的排序质量越好。
    IDCG = np.array([(tp[:min(n, k)]).sum() for n in heldout_batch.sum(axis=1).astype(np.int32)])
    return DCG / IDCG


def Recall_Precision_F1_OneCall_at_k_batch(X_pred, heldout_batch, k=5):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
    X_true_binary = (heldout_batch > 0)
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(np.float32)
    recall = tmp / X_true_binary.sum(axis=1)
    precision = tmp / k
    f1 = 2 * recall * precision / (recall + precision)
    oneCall = (tmp > 0).astype(np.float32)
    return recall, precision, f1, oneCall


def AUC_at_k_batch(X_train, X_pred, heldout_batch):
    train_set_num = X_train.sum(axis=1)
    test_set_num = heldout_batch.sum(axis=1)
    sorted_id = np.argsort(X_pred, axis=1)
    rank = np.argsort(sorted_id) + 1
    molecular = (heldout_batch * rank).sum(axis=1) - test_set_num * (
                test_set_num + 1) / 2 - test_set_num * train_set_num
    denominator = (X_pred.shape[1] - train_set_num - test_set_num) * test_set_num
    aucs = molecular / denominator
    return aucs

# 计算模型的大小
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
def getModelCal(model, batch_size, user_num, item_num):
    # 打印模型的摘要信息，包括参数量和计算量
    # TODO: summary函数的输入参数不正确！
    summary(model, input_size=[(batch_size, user_num), (batch_size, item_num)])

if __name__ == '__main__':
    # 示例数据
    scores = torch.tensor([4.5, 2.3, 3.8, 1.9, 4.1])
    ratings = torch.tensor([5, 2, 4, 1, 3])

    # 计算 NDCG 值
    k = 3  # Top-K 位置
    ndcg = calculate_ndcg(scores, ratings, k)

    print("NDCG@{}: {:.4f}".format(k, ndcg))
