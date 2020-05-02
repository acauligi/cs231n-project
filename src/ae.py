"""
Autoencoder baseline
"""

import torch
from torch import nn
from losses import kl_bernoulli


class AE(nn.Module):
    def __init__(self, dim_in, dim_z, config='pendulum'):
        super(AE, self).__init__()
        _, _, dec = load_config(config)

        # TODO, refactor encoder to allow output of dim_z instead of dim_z * 2
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Linear(800, dim_z),
            nn.BatchNorm1d(dim_z),
            nn.Sigmoid()
        )

        self.decoder = dec(dim_z, dim_in)

    def forward(self, x):
        self.z = self.encoder(x)
        return self.decoder(self.z)


def compute_loss(x_pred, x_true, z_pred, z_true, beta=0.05):
    mse = nn.MSELoss()
    return mse(x_pred, x_true).add(beta * kl_bernoulli(z_pred, z_true))
