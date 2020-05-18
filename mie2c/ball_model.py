from mie2c.e2c import Encoder, Decoder, Transition

import torch
from torch import nn


def get_ball_encoder(dim_in, dim_z): 
    channels_enc = [3, 32, 32]
    ff_shape = [512, 512]

    conv_activation = torch.nn.ReLU()
    ff_activation = torch.nn.ReLU()

    n_channels = len(channels_enc) - 1
    kernel_enc = [2,3]
    stride= [2] * n_channels
    padding= [2] * n_channels
    pool = [2, 2] * n_channels

    return Encoder(dim_in, dim_z, channels_enc, ff_shape, kernel_enc, stride, padding, pool, conv_activation=conv_activation, ff_activation=ff_activation)


def get_ball_decoder(dim_in, dim_out): 
    channels_dec = [32, 32, dim_out[0]]
    ff_shape = [512, 512]

    conv_activation = torch.nn.ReLU()
    ff_activation = torch.nn.ReLU()

    n_channels = len(channels_dec) - 1
    kernel_dec = [2, 2]
    stride= [2] * n_channels
    padding= [2] * n_channels
    pool = [2, 2] * n_channels

    return Decoder(dim_in, dim_out, channels_dec, ff_shape, kernel_dec, stride, padding, ff_activation=ff_activation, conv_activation=conv_activation)


def get_ball_transition(dim_z, dim_u):
    trans = nn.Sequential(
        nn.Linear(dim_z, 100),
        nn.BatchNorm1d(100),
        nn.ReLU(),
        nn.Linear(100, 100),
        nn.BatchNorm1d(100),
        nn.ReLU(),
        nn.Linear(100, dim_z*2)
    )

    return Transition(trans, dim_z, dim_u)