import torch
from model import RegLSTM, GRU
from torch import optim
from utils import mae, metrics


class LSTM_trainer:
    def __init__(self, device, inp_dim, out_dim, mid_layers, lr, weight_decay, decay):
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.model = RegLSTM(inp_dim, out_dim, 128, mid_layers)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss_fn = mae
        self.clip = None

    def train(self, inp, label):
        self.model.train()
        self.optimizer.zero_grad()
        inp = inp.view([-1, 1, self.inp_dim])
        label = label.view([-1, 1, self.out_dim])
        pred = self.model(inp)
        loss = self.loss_fn(pred, label)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae_error, rmse_error = metrics(pred, label)
        return loss.item(), mae_error, rmse_error

    def eval(self, inp, label):
        self.model.eval()
        inp = inp.view([-1, 1, self.inp_dim])
        label = label.view([-1, 1, self.out_dim])
        pred = self.model(inp)
        loss = self.loss_fn(pred, label)
        mae_error, rmse_error = metrics(pred, label)
        return loss.item(), mae_error, rmse_error


class GRU_trainer:
    def __init__(self, device, inp_dim, out_dim, lr, weight_decay, decay):
        self.inp_dim, self.out_dim = inp_dim, out_dim
        self.model = GRU(inp_dim, out_dim, 512)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, decay)
        self.loss_fn = mae
        self.clip = None

    def train(self, inp, label):
        self.model.train()
        self.optimizer.zero_grad()
        inp = inp.view([-1, 1, self.inp_dim])
        label = label.view([-1, 1, self.out_dim])
        pred = self.model(inp)
        loss = self.loss_fn(pred, label)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        mae_error, rmse_error = metrics(pred, label)
        return loss.item(), mae_error, rmse_error

    def eval(self, inp, label):
        self.model.eval()
        inp = inp.view([-1, 1, self.inp_dim])
        label = label.view([-1, 1, self.out_dim])
        pred = self.model(inp)
        loss = self.loss_fn(pred, label)
        mae_error, rmse_error = metrics(pred, label)
        return loss.item(), mae_error, rmse_error
