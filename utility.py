import math
import os
import random
import time as time

import numpy as np
import scipy
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from Log import log_print


def extract_axis_1(data, indices):
    res = []
    for i in range(data.shape[0]):
        res.append(data[i, indices[i], :])
    res = torch.stack(res, dim=0).unsqueeze(1)
    return res


def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + ".df"))


def pad_history(itemlist, length, pad_item):
    if len(itemlist) >= length:
        return itemlist[-length:]
    if len(itemlist) < length:
        temp = [pad_item] * (length - len(itemlist))
        itemlist.extend(temp)
        return itemlist


def calculate_hit(sorted_list, topk, true_items, hit_purchase, ndcg_purchase):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i] :]

        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])

                hit_purchase[i] += 1.0
                ndcg_purchase[i] += 1.0 / np.log2(rank + 1)


def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = []
    recall = []
    NDCG = []
    MRR = []

    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):
            if GroundTruth[i] is not None:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0 / math.log2(j + 2)
                        if mrrFlag:
                            userMRR = 1.0 / (j + 1.0)
                            mrrFlag = False
                        userHit += 1

                    if idcgCount > 0:
                        idcg += 1.0 / math.log2(j + 2)
                        idcgCount = idcgCount - 1

                if idcg != 0:
                    ndcg += dcg / idcg

                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])
                sumForNdcg += ndcg
                sumForMRR += userMRR

        precision.append(round(sumForPrecision / len(predictedIndices), 4))
        recall.append(round(sumForRecall / len(predictedIndices), 4))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
        MRR.append(round(sumForMRR / len(predictedIndices), 4))

    return precision, recall, NDCG, MRR


def print_results(loss=None, test_result=None):
    """output the evaluation results."""
    if loss is not None:
        log_print("[Train]: loss: {:.4f}".format(loss))
    log_print(
        "[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
            "-".join(["{:.4f}".format(x) for x in test_result["Precision"]]),
            "-".join(["{:.4f}".format(x) for x in test_result["Recall"]]),
            "-".join(["{:.4f}".format(x) for x in test_result["NDCG"]]),
            "-".join(["{:.4f}".format(x) for x in test_result["MRR"]]),
        )
    )


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_adj(data_directory, data_name, shape):
    adj = np.load(os.path.join(data_directory, data_name))
    users = adj[:, 0]
    items = adj[:, 1]
    data = np.ones(len(users))
    adj = scipy.sparse.csr_matrix((data, (users, items)), shape)
    return adj


def get_num_users_items(data_directory, data_name):
    adj = np.load(os.path.join(data_directory, data_name))
    users = adj[:, 0]
    items = adj[:, 1]
    return max(users), max(items)


def sampling_neg(user, pos, train_dataset):
    all_pos = train_dataset.all_pos 
    negs = np.zeros(len(user), dtype=np.int32)
    for idx, u in enumerate(user):
        user_pos_set = set(all_pos[u.item()])

        while True:
            neg_index = torch.randint(0, len(pos), (1,)).item()

            if pos[neg_index] not in user_pos_set:
                negs[idx] = neg_index
                break  

    return negs


def target_item_gen(adj):
    target_data = {}
    rows, cols = adj.nonzero()
    for i, item in enumerate(cols):
        user = rows[i]
        if target_data.get(user):
            target_data[user].append(item)
        else:
            target_data[user] = [item]
    return target_data


def compute_prediction(
    user_emb_tmp, item_emb_tmp, n_layers, UIgraph, n_user, n_item, tst_users, model
):
    all_emb = torch.cat([user_emb_tmp, item_emb_tmp])
    embs = [all_emb]
    for layer in range(n_layers):
        all_emb = torch.sparse.mm(UIgraph, all_emb)
        embs.append(all_emb)
    embs = torch.stack(embs, dim=1)
    # print(embs.size())
    light_out = torch.mean(embs, dim=1)
    users_final, items_final = torch.split(light_out, [n_user, n_item])
    pred = torch.matmul(users_final[tst_users], items_final.t())
    pred = model.f(pred)
    return pred


def save_best_result(result, data, name):
    filename = "./result/" + data + "/" + name + ".txt"
    if not os.path.exists(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as file:
        labels = ["Precision", "Recall", "NDCG", "MRR"]
        for label, values in zip(labels, result):
            line = label + " " + " ".join(f"{value}" for value in values)
            file.write(line + "\n")


def getSparseGraph(data_directory, n_user, n_item, adj_trn, device):
    log_print("loading adjacency matrix")
    try:
        pre_adj_mat = sp.load_npz(data_directory + "/s_pre_adj_mat.npz")
        log_print("successfully loaded...")
        norm_adj = pre_adj_mat
    except:
        log_print("generating adjacency matrix")
        s = time.time()
        adj_mat = sp.dok_matrix((n_user + n_item, n_user + n_item), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = adj_trn.tocsr().tolil()
        adj_mat[:n_user, n_user:] = R
        adj_mat[n_user:, :n_user] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        end = time.time()
        log_print(f"costing {end - s}s, saved norm_mat...")
        sp.save_npz(data_directory + "/s_pre_adj_mat.npz", norm_adj)

    Graph = _convert_sp_mat_to_sp_tensor(norm_adj)
    Graph = Graph.coalesce().to(device)
    log_print("don't split the matrix")
    return Graph


def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

def calc_ssl_loss_1(
    pos_item_list, item_view1, item_emb, tau=0.5
):
    i_emd1 = F.normalize(item_view1, dim=1)
    i_emd2 = F.normalize(item_emb[pos_item_list], dim=1)
    all_item2 = F.normalize(item_emb, dim=1)

    v_pos = torch.sum(i_emd1 * i_emd2, dim=1)

    v_all = i_emd1.matmul(all_item2.T)

    v_pos = torch.exp(v_pos / tau)
    v_all = torch.sum(torch.exp(v_all / tau), dim=1)

    ssl_item = -torch.sum(torch.log(v_pos / v_all))
    return ssl_item

def calc_ssl_loss_2(
    item_emb1, item_emb2, tau=0.5
):

    item_emb1 = F.normalize(item_emb1, dim=1)
    item_emb2 = F.normalize(item_emb2, dim=1)

    sim_matrix = item_emb1 @ item_emb2.T  # [B, B]
    sim_matrix /= tau

    pos_score = torch.exp(torch.diag(sim_matrix))  # [B]

    all_scores = torch.exp(sim_matrix).sum(dim=1)  # [B]

    loss = -torch.sum(torch.log(pos_score / all_scores))
    return loss