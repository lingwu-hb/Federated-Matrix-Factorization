from metric import *

def evaluateModel(model, epoch, testDataSet, device):
    # evaluate phase
    ndcg5_list = []
    recall5_list = []
    precision5_list = []
    f1_list = []
    oneCall_list = []
    auc_list = []

    model.eval()
    with torch.no_grad():
        for batch in testDataSet:
            batch_dict = dict([(k, v[0].float().to(device)) for k, v in batch.items()])
            users = batch_dict['user'].int()
            items = batch_dict['item'].int()
            ratings = batch_dict['ratings']

            # scores -> tensor(20): 一维tensor
            scores = model({'user': users, 'item': items})

            n_5 = calculate_ndcg(scores, ratings, 5)
            r_5, p_5, f_5, o_5 = calculate_Recall_Preision_F1_OneCall(scores, ratings, 5)
            auc_b = calculate_AUC_at_k(scores, ratings, 5)

            ndcg5_list.append(np.atleast_1d(n_5))
            recall5_list.append(np.atleast_1d(r_5))
            precision5_list.append(np.atleast_1d(p_5))
            f1_list.append(np.atleast_1d(f_5))
            oneCall_list.append(np.atleast_1d(o_5))
            auc_list.append(np.atleast_1d(auc_b))

    ndcg5_list = np.concatenate(ndcg5_list)
    recall5_list = np.concatenate(recall5_list)
    precision5_list = np.concatenate(precision5_list)
    f1_list = np.concatenate(f1_list)
    oneCall_list = np.concatenate(oneCall_list)
    auc_list = np.concatenate(auc_list)

    ndcg5_list[np.isnan(ndcg5_list)] = 0
    ndcg5_mean = np.mean(ndcg5_list)
    ndcg5_var = np.var(ndcg5_list)
    recall5_list[np.isnan(recall5_list)] = 0
    recall5_mean = np.mean(recall5_list)
    recall5_var = np.var(recall5_list)
    precision5_list[np.isnan(precision5_list)] = 0
    precision5_mean = np.mean(precision5_list)
    precision5_var = np.var(precision5_list)
    f1_list[np.isnan(f1_list)] = 0
    f1_mean = np.mean(f1_list)
    f1_var = np.var(f1_list)
    oneCall_list[np.isnan(oneCall_list)] = 0
    oneCAll_mean = np.mean(oneCall_list)
    oneCAll_var = np.var(oneCall_list)
    auc_list[np.isnan(auc_list)] = 0
    auc_mean = np.mean(auc_list)
    auc_var = np.var(auc_list)

    print(
        "Epoch: {:3d} | Pre@5: {:5.4f} | Rec@5: {:5.4f} | F1@5: {:5.4f} | NDCG@5: {:5.4f} | 1-call@5: {:5.4f} | AUC: {:5.4f}".format(
            epoch + 1, precision5_mean, recall5_mean, f1_mean, ndcg5_mean, oneCAll_mean, auc_mean), flush=True)

    print(
        "Epoch: {:3d} | PreV@5: {:5.4f} | RecV@5: {:5.4f} | F1V@5: {:5.4f} | NDCGV@5: {:5.4f} | 1-callV@5: {:5.4f} | AUCV: {:5.4f}".format(
            epoch + 1, precision5_var, recall5_var, f1_var, ndcg5_var, oneCAll_var, auc_var), flush=True)

    return ndcg5_mean