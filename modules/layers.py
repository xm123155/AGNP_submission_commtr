import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, neuron_list):
        super().__init__()
        network_size_list = [input_size] + neuron_list
        network_list = []
        for i in range(1, len(network_size_list) - 1):
            network_list.append(nn.Linear(network_size_list[i-1], network_size_list[i], bias=True))
            network_list.append(nn.ReLU())
        network_list.append(nn.Linear(network_size_list[-2], network_size_list[-1]))
        self.mlp = nn.Sequential(*network_list)

    def forward(self, x):
        return self.mlp(x)


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, inp, adj):
        inp = F.dropout(inp, self.dropout, self.training)
        support = torch.mm(inp, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output


# Qin, S., Zhu, J., Qin, J., Wang, W., & Zhao, D. (2019). Recurrent attentive neural process for sequential data.
# arXiv preprint arXiv:1910.09323.  https://github.com/3springs/attentive-neural-processes
class AttnLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        torch.nn.init.normal_(self.linear.weight, std=in_channels ** -0.5)

    def forward(self, x):
        x = self.linear(x)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_dim, attention_type, n_heads=8, x_dim=1, rep="mlp", mlp_hidden_dim_list=[]):
        super().__init__()
        self._rep = rep

        if self._rep == "mlp":
            self.mlp_k = MLP(x_dim, mlp_hidden_dim_list)
            self.mlp_q = MLP(x_dim, mlp_hidden_dim_list)

        if attention_type == "dot":
            self._attention_func = self._dot_attention
        elif attention_type == "multihead":
            self._W_k = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W_v = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W_q = nn.ModuleList(
                [AttnLinear(hidden_dim, hidden_dim) for _ in range(n_heads)]
            )
            self._W = AttnLinear(n_heads * hidden_dim, hidden_dim)
            self._attention_func = self._multihead_attention
            self.n_heads = n_heads
        else:
            raise NotImplementedError

    def forward(self, k, v, q):
        if self._rep == "mlp":
            k = self.mlp_k(k)
            q = self.mlp_q(q)
        rep = self._attention_func(k, v, q)
        return rep

    def _dot_attention(self, k, v, q):
        scale = q.shape[-1] ** 0.5
        unnorm_weights = torch.einsum("bjk,bik->bij", k, q) / scale
        weights = torch.softmax(unnorm_weights, dim=-1)

        rep = torch.einsum("bik,bkj->bij", weights, v)
        return rep

    def _multihead_attention(self, k, v, q):
        outs = []
        for i in range(self.n_heads):
            k_ = self._W_k[i](k)
            v_ = self._W_v[i](v)
            q_ = self._W_q[i](q)
            out = self._dot_attention(k_, v_, q_)
            outs.append(out)
        outs = torch.stack(outs, dim=-1)
        outs = outs.view(outs.shape[0], outs.shape[1], -1)
        rep = self._W(outs)
        return rep
