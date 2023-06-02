import torch
from torch import nn
import torch.nn.functional as F
from .layers import MLP


class Decoder(nn.Module):
    def __init__(self, x_dim, neuron_list, hidden_dim):
        super(Decoder, self).__init__()
        self.mlp = MLP(x_dim + 2 * hidden_dim, neuron_list + [2])

    def forward(self, r, z, target_x):
        inp = torch.cat([r, z, target_x], dim=-1)
        mu_sigma = self.mlp(inp)
        mu, log_sigma = mu_sigma.chunk(chunks=2, dim=-1)
        mu, log_sigma = mu.mean(dim=0).squeeze(), log_sigma.mean(dim=0).squeeze()
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)
        distribution = torch.distributions.Normal(mu, sigma)
        return distribution, mu, sigma
