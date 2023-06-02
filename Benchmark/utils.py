import pickle
import numpy as np
import torch


class DataLoader:
    def __init__(self, x, y, batch_size):
        self.batch_size = batch_size
        self.current_id = 0
        self.num_sample = len(x)
        self.num_batch = int(self.num_sample // self.batch_size)
        self.x = x
        self.y = y

    def shuffle(self):
        permutation = np.random.permutation(self.num_sample)
        self.x = self.x[permutation]
        self.y = self.y[permutation]

    def get_iterator(self):
        self.current_id = 0

        def _wrapper():
            while self.current_id < self.num_batch:
                start_id = self.batch_size * self.current_id
                end_id = min(self.num_sample, self.batch_size * (self.current_id + 1))
                x_batch = self.x[start_id: end_id, ...]
                y_batch = self.y[start_id: end_id, ...]
                yield (x_batch, y_batch)
                self.current_id += 1

        return _wrapper()


def mae(pred, label):
    loss = torch.abs(pred - label)
    return torch.mean(loss)


def rmse(pred, label):
    loss = (pred - label) ** 2
    return torch.sqrt(torch.mean(loss))


def metrics(pred, label):
    mae_loss = mae(pred, label).item()
    rmse_loss = rmse(pred, label).item()
    return mae_loss, rmse_loss


def load_dataset_agnp(data_path, batch_size, percent):
    data = dict()
    datasets = []
    for i in range(4500):
        if percent == 10:
            dataset = pickle.load(open(data_path + '/sub_data_of_time_{}.pkl'.format(i), 'rb'))
        else:
            dataset = pickle.load(open(data_path + '/sub_data_of_time_{}_{}%.pkl'.format(i, percent), 'rb'))
        datasets.append(dataset)
    datasets = np.concatenate(datasets, axis=0)
    data['x_train'], data['y_train'] = datasets[:, :12], datasets[:, 12]
    datasets = []
    for i in range(4500, 4680):
        dataset = pickle.load(open(data_path + '/data_of_time_{}.pkl'.format(i), 'rb'))
        datasets.append(dataset)
    datasets = np.concatenate(datasets, axis=0)
    data['x_test'], data['y_test'] = datasets[:, :12], datasets[:, 12]
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], batch_size)
    return data
