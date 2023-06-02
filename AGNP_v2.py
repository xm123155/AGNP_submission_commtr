# -*- coding: utf-8 -*-
# Kim, H., Mnih, A., Schwarz, J., Garnelo, M., Eslami, A., Rosenbaum, D., ... & Teh, Y. W. (2019). Attentive neural
# processes. arXiv preprint arXiv:1901.05761.  https://github.com/deepmind/neural-processes
# Qin, S., Zhu, J., Qin, J., Wang, W., & Zhao, D. (2019). Recurrent attentive neural process for sequential data.
# arXiv preprint arXiv:1910.09323.  https://github.com/3springs/attentive-neural-processes

from __future__ import division
from __future__ import print_function
from args import args
import time
import numpy as np
import torch
from modules.model import Graph_Encoder
from modules.optimizer import loss_function2
from utils import preprocess_graph, mask_graph, load_data_agnp
from modules.anp import Anp

torch.set_default_dtype(torch.float32)


def train():
    # full_adj_set: overall road speed; sub_adj_set: full_adj_set minus random sensors and roads
    full_adj_set, sub_adj_set, features, targets = load_data_agnp(10)
    print("Using {} dataset".format(args.dataset_str))
    n_nodes, feat_dim, = features[-1].shape
    graph_encoder = Graph_Encoder(feat_dim, args.hiddenEnc, args.dropout).to(args.device)
    anp_model = Anp(128, [256, 256], 256).to(args.device)
    optimizer = torch.optim.Adam(list(graph_encoder.parameters()) + list(anp_model.parameters()), args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.995)

    train_loss = []
    for epoch in range(args.epochs):
        time_index = np.random.randint(4500)
        t = time.time()
        graph_encoder.train()
        anp_model.train()
        adj_context = mask_graph(sub_adj_set[time_index:time_index + 12], 10)
        adj_context_norm = preprocess_graph(adj_context).to(args.device)
        adj_train_norm = preprocess_graph(sub_adj_set[time_index:time_index + 12]).to(args.device)
        target = torch.FloatTensor(targets[time_index + 192]).to(args.device)
        context_features = features[time_index:time_index + 12]

        context_x = graph_encoder(context_features, adj_context_norm)
        target_x = graph_encoder(context_features, adj_train_norm)
        mu, sigma, kl = anp_model(context_x=context_x, target_x=target_x)

        loss, mae, mse = loss_function2(pred=[mu, sigma], labels=target, kl=kl)

        optimizer.zero_grad()
        (loss + mae).backward()
        cur_loss = loss.item()
        torch.nn.utils.clip_grad_norm_(list(graph_encoder.parameters()) + list(anp_model.parameters()), args.max_norm)
        optimizer.step()
        scheduler.step()
        train_loss.append(cur_loss)
        print("Epoch:", '%04d' % epoch, "train_loss=", "{:.5f}".format(cur_loss),
              "time=", "{:.5f}".format(time.time() - t), 'mae={:.5f}'.format(mae.item()),
              'rmse={:.5f}'.format(np.sqrt(mse.item())))

        if (epoch + 1) % 10000 == 0:
            graph_encoder.eval()
            anp_model.eval()
            optimizer.zero_grad()
            mae_list, mse_list = [], []
            with torch.no_grad():
                for time_index in range(4500, 4680):
                    adj_context = mask_graph(sub_adj_set[time_index:time_index + 12], 10)
                    adj_context_norm = preprocess_graph(adj_context).to(args.device)
                    adj_train_norm = preprocess_graph(sub_adj_set[time_index:time_index + 12]).to(args.device)
                    context_features = features[time_index:time_index + 12]
                    target = torch.FloatTensor(targets[time_index + 192]).to(args.device)

                    context_x = graph_encoder(context_features, adj_context_norm)
                    target_x = graph_encoder(context_features, adj_train_norm)
                    mu, sigma, kl = anp_model(context_x=context_x, target_x=target_x)
                    loss, mae, mse = loss_function2(pred=[mu, sigma], labels=target, kl=kl)
                    mae_list.append(mae.item())
                    mse_list.append(mse.item())
                print('test mae={:.5f}'.format(np.mean(mae_list)), 'rmse={:.5f}'.format(np.sqrt(np.mean(mse_list))))
                print(mae_list)
                print(mse_list)
                torch.save(graph_encoder.state_dict(), 'logs/epoch_{}_graph.pth'.format(epoch))
                torch.save(anp_model.state_dict(), 'logs/epoch_{}_anp.pth'.format(epoch))


def test():
    # full_adj_set: overall road speed; sub_adj_set: full_adj_set minus random sensors and roads
    full_adj_set, sub_adj_set, features, targets = load_data_agnp(10)
    print("Using {} dataset".format(args.dataset_str))
    n_nodes, feat_dim, = features[-1].shape
    graph_encoder = Graph_Encoder(feat_dim, args.hiddenEnc, args.dropout).to(args.device)
    anp_model = Anp(128, [256, 256], 256).to(args.device)
    state = torch.load('logs/epoch_graph.pth', map_location='cuda')
    graph_encoder.load_state_dict(state)
    state = torch.load('logs/epoch_anp.pth', map_location='cuda')
    anp_model.load_state_dict(state)
    graph_encoder.eval()
    anp_model.eval()
    mae_list, mse_list = [], []
    with torch.no_grad():
        for time_index in range(4500, 4680):
            adj_context = mask_graph(sub_adj_set[time_index:time_index + 12], 10)
            adj_context_norm = preprocess_graph(adj_context).to(args.device)
            adj_train_norm = preprocess_graph(sub_adj_set[time_index:time_index + 12]).to(args.device)
            context_features = features[time_index:time_index + 12]
            target = torch.FloatTensor(targets[time_index + 192]).to(args.device)

            context_x = graph_encoder(context_features, adj_context_norm)
            target_x = graph_encoder(context_features, adj_train_norm)
            mu, sigma, kl = anp_model(context_x=context_x, target_x=target_x)
            loss, mae, mse = loss_function2(pred=[mu, sigma], labels=target, kl=kl)
            mae_list.append(mae.item())
            mse_list.append(mse.item())
        print('test mae={:.5f}'.format(np.mean(mae_list)), 'rmse={:.5f}'.format(np.sqrt(np.mean(mse_list))))
        print(mae_list)
        print(mse_list)


if __name__ == '__main__':
    if args.train:
        train()
    else:
        test()
