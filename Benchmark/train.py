import config
from trainer import LSTM_trainer, GRU_trainer
from utils import load_dataset_agnp
import torch
import time
import numpy as np
import pickle as pkl


def data_reader(percent):
    if config.dataset == 'agnp':
        data_path = ''
        dataloader = load_dataset_agnp(data_path, config.batch_size, percent)
    else:
        raise ValueError('no such dataset')
    return dataloader


def train(dataloader, trainer, percent):
    total_train_loss = []
    for epoch in range(1, config.epoch + 1):
        start = time.time()
        train_loss, train_mae, train_rmse = [], [], []
        dataloader['train_loader'].shuffle()
        for i, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            train_x = torch.Tensor(x).to(config.device)
            train_y = torch.Tensor(y).to(config.device)
            metric = trainer.train(train_x, train_y)
            train_loss.append(metric[0])
            train_mae.append(metric[1])
            train_rmse.append(metric[2])
        avg_train_loss = np.mean(train_loss)
        avg_train_mae = np.mean(train_mae)
        avg_train_rmse = np.sqrt(np.mean(np.array(train_rmse) ** 2))
        total_train_loss.append(avg_train_loss)
        end = time.time()
        log = '{} Epoch: {:03d}, Train Loss: {:.4f}, Train MAE: {:.4f}, Train RMSE: {:.4f}, Training time: {:.4f}'
        print(log.format(config.model, epoch, avg_train_loss, avg_train_mae, avg_train_rmse, round(end - start, 3),
                         flush=True))
        torch.save(trainer.model.state_dict(), 'logs/{}_{}_{}'.format(config.model, epoch, percent))

        test_loss, test_mae, test_rmse, = [], [], []
        for i, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
            test_x = torch.Tensor(x).to(config.device)
            test_y = torch.Tensor(y).to(config.device)
            metric = trainer.eval(test_x, test_y)
            test_loss.append(metric[0])
            test_mae.append(metric[1])
            test_rmse.append(metric[2])
        avg_test_mae = np.mean(test_mae)
        avg_test_rmse = np.sqrt(np.mean(np.array(test_rmse) ** 2))
        log = 'Test MAE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(avg_test_mae, avg_test_rmse))


def test(dataloader, trainer, percent):
    trainer.model.load_state_dict(torch.load('logs/{}_{}'.format(config.model, percent)))
    test_loss, test_mae, test_rmse, = [], [], []
    for i, (x, y) in enumerate(dataloader.get_iterator()):
        test_x = torch.Tensor(x).to(config.device)
        test_y = torch.Tensor(y).to(config.device)
        metric = trainer.eval(test_x, test_y)
        test_loss.append(metric[0])
        test_mae.append(metric[1])
        test_rmse.append(metric[2])
    avg_test_mae = np.mean(test_mae)
    avg_test_rmse = np.sqrt(np.mean(np.array(test_rmse) ** 2))
    log = 'Test MAE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(avg_test_mae, avg_test_rmse))


def main():
    percent = config.percent
    if config.model == 'lstm':
        trainer = LSTM_trainer(config.device, config.inp_dim, config.out_dim, config.mid_layers,
                               config.lr, config.weight_decay, config.decay)
    elif config.model == 'gru':
        trainer = GRU_trainer(config.device, config.inp_dim, config.out_dim, config.lr,
                              config.weight_decay, config.decay)
    else:
        raise ValueError('no such model')

    if config.dataset == 'agnp':
        if config.train:
            dataloader = data_reader(percent)
            train(dataloader, trainer, percent)
        else:
            dataloader = pkl.load(open('data/data_loader_{}.pkl'.format(percent), 'rb'))
            test(dataloader, trainer, percent)
    else:
        raise ValueError('no such model')


if __name__ == '__main__':
    main()
