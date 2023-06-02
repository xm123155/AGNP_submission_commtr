import torch
from torch import nn
from .layers import Attention


class LatentPath(nn.Module):
    def __init__(self, x_dim, hidden_dim, self_attn_type):
        super().__init__()
        self.self_attn = Attention(hidden_dim, self_attn_type, rep='identity')
        self.l1 = nn.Linear(x_dim, x_dim)
        self.l2_mu = nn.Linear(x_dim, hidden_dim)
        self.l2_std = nn.Linear(x_dim, hidden_dim)

    def forward(self, x):
        s_i = self.self_attn(x, x, x)
        s_c = s_i.mean(dim=1)
        z = torch.relu(self.l1(s_c))
        mu = self.l2_mu(z)
        log_sigma = self.l2_std(z)
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)

        return torch.distributions.Normal(mu, sigma), mu, sigma
