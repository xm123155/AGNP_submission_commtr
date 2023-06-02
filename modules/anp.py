from torch import nn
import torch
from modules.latent_encoder import LatentPath
from modules.deterministic_encoder import DeterministicPath
from modules.decoder import Decoder


class Anp(nn.Module):
    def __init__(self, x_dim, neuron_list, hidden_dim, self_attn_type="dot", cross_attn_type="multihead"):
        super(Anp, self).__init__()
        self.latent = LatentPath(x_dim, hidden_dim, self_attn_type)
        self.decoder = Decoder(x_dim, neuron_list, hidden_dim)
        self.deterministic = DeterministicPath(x_dim, neuron_list, hidden_dim, self_attn_type, cross_attn_type)

    def forward(self, context_x, target_x):
        prior_distribution, prior_mu, prior_sigma = self.latent(context_x)
        post_distribution, post_mu, post_sigma = self.latent(target_x)
        z = post_distribution.loc.unsqueeze(1).repeat(1, target_x.size()[1], 1)
        r = self.deterministic(context_x, target_x)
        distribution, mu, sigma = self.decoder(r, z, target_x)
        kl = torch.distributions.kl_divergence(post_distribution, prior_distribution).mean()
        return mu, sigma, kl
