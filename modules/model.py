import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution, MLP, Attention
from args import args


class Graph_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim1, dropout, hidden_dim2=128):
        super(Graph_Encoder, self).__init__()
        self.MLP = MLP(input_dim, [hidden_dim1])
        self.gc1 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim2, hidden_dim2, dropout, act=lambda x: x)
        self.MLP2 = MLP(hidden_dim2, [hidden_dim2])
        self.relu = torch.nn.ReLU()
        self.attention = Attention(hidden_dim2, attention_type='multihead', rep='identity')

    def forward(self, x, adj):
        representations = []
        for i in range(len(adj)):
            hidden1 = self.MLP(x[i].to(args.device))
            hidden1 = self.relu(hidden1)
            hidden2 = self.gc1(hidden1, adj[i])
            hidden3 = self.gc2(hidden2, adj[i])
            hidden4 = self.MLP2(hidden3)
            representations.append(hidden4)
        representations = torch.stack(representations)
        return representations
