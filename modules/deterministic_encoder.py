from torch import nn
from .layers import MLP, Attention


class DeterministicPath(nn.Module):
    def __init__(self, x_dim, neuron_list, hidden_dim, self_attn_type="dot", cross_attn_type="multihead"):
        super().__init__()
        self.mlp = MLP(x_dim, neuron_list)
        self.self_attn = Attention(hidden_dim, self_attn_type, rep='identity')
        self.cross_attn = Attention(hidden_dim, cross_attn_type, x_dim=x_dim, mlp_hidden_dim_list=neuron_list,
                                    rep='mlp')

    def forward(self, context_x, target_x):
        r_i = self.mlp(context_x)
        r_i = self.self_attn(r_i, r_i, r_i)
        r = self.cross_attn(context_x, r_i, target_x)
        return r
