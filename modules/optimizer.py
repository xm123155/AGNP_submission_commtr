import torch
from args import args
import pickle as pkl

connectedness = pkl.load(open(args.connectedness_matrix, 'rb'))


def metrics(pred, label, num):
    mae_loss = mae(pred, label, num)
    mse_loss = mse(pred, label, num)
    return mae_loss, mse_loss


def mae(pred, label, num):
    loss = torch.abs(pred - label)
    return torch.sum(loss) / num


def mse(pred, label, num):
    loss = (pred - label) ** 2
    return torch.sum(loss) / num


def loss_function2(pred, labels, kl):
    [mu, std] = pred
    pred = torch.distributions.Normal(mu, std)
    log_p = pred.log_prob(labels).mean()
    mae_loss, mse_loss = metrics(pred.loc, labels, len(mu))
    return -log_p + kl, mae_loss, mse_loss
