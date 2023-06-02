import pickle as pkl
import numpy as np
import scipy.sparse as sp
import torch


def load_data_agnp(percent):
    full_adj_set = pkl.load(open('data/full_adj_set_{}.pkl'.format(percent), 'rb'))
    sub_adj_set = pkl.load(open('data/sub_adj_set_{}.pkl'.format(percent), 'rb'))
    features = pkl.load(open('data/features_{}.pkl'.format(percent), 'rb'))
    target = pkl.load(open('data/target_{}.pkl'.format(percent), 'rb'))
    return full_adj_set, sub_adj_set, features, target


def mask_graph(adj, percent):
    data_0 = pkl.load(open('data/data_0.pkl', 'rb'))
    if percent == 10:
        random_road = np.sort(np.random.randint(0, 348, np.random.randint(15, 25)))
        random_sensor = np.sort(np.random.randint(0, 1196, np.random.randint(10, 30)))
    elif percent == 40:
        random_road = np.sort(np.random.randint(0, 348, np.random.randint(15, 25)))
        random_sensor = np.sort(np.random.randint(0, 1196, np.random.randint(10, 30)))
    elif percent == 70:
        random_road = np.sort(np.random.randint(0, 348, np.random.randint(15, 25)))
        random_sensor = np.sort(np.random.randint(0, 1196, np.random.randint(10, 30)))
    else:
        return ValueError
    adj_train = []
    for i in adj:
        data_5min = i.toarray()
        for j in random_road:
            if j == 0:
                from_0, to_0 = 0, data_0[j]
            else:
                from_0, to_0 = data_0[j - 1] + 1, data_0[j]
            for k in range(from_0, to_0 + 1):
                data_5min[k, :] = np.zeros(len(data_5min))
        for j in random_sensor:
            data_5min[j, :] = np.zeros(len(data_5min))
        for j in range(len(data_5min)):
            data_5min[j, j] = max(np.max(data_5min[j, :]), 1)
        data_5min = sp.csr_matrix(data_5min)
        adj_train.append(data_5min)
    return adj_train


def preprocess_graph(adj):
    processed = []
    for i in adj:
        i = sp.coo_matrix(i)
        rowsum = np.array(i.sum(1))
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
        adj_normalized = i.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        processed.append(mx_tensor(adj_normalized))
    processed = torch.stack(processed)
    return processed


def mx_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
