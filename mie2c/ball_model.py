from mie2c.e2c import Encoder, Decoder, Transition, LinearTransition, PWATransition

import torch
from torch import nn


def get_ball_encoder(dim_in, dim_z):
    channels_enc = [dim_in[0], 16, 16]
    ff_shape = [256, 256]

    conv_activation = torch.nn.ReLU()
    ff_activation = torch.nn.ReLU()

    n_channels = len(channels_enc) - 1
    kernel_enc = [2, 3]
    stride = [2] * n_channels
    padding = [2] * n_channels
    pool = [2, 2] * n_channels

    return Encoder(dim_in, dim_z, channels_enc, ff_shape, kernel_enc, stride, padding, pool, conv_activation=conv_activation, ff_activation=ff_activation)


def get_ball_decoder(dim_in, dim_out):
    channels_dec = [16, 16, dim_out[0]]
    ff_shape = [256, 256]

    conv_activation = torch.nn.ReLU()
    ff_activation = torch.nn.ReLU()

    n_channels = len(channels_dec) - 1
    kernel_dec = [2, 2]
    stride = [2] * n_channels
    padding = [2] * n_channels

    return Decoder(dim_in, dim_out, channels_dec, ff_shape, kernel_dec, stride, padding, ff_activation=ff_activation, conv_activation=conv_activation)


def get_ball_transition(dim_z, dim_u):
    nn_width = 20
    trans = nn.Sequential(
        nn.Linear(dim_z, nn_width),
        nn.BatchNorm1d(nn_width),
        nn.ReLU(),
        nn.Linear(nn_width, nn_width),
        nn.BatchNorm1d(nn_width),
        nn.ReLU(),
        nn.Linear(nn_width, dim_z*2)
    )

    return Transition(trans, dim_z, dim_u)


def get_ball_linear_transition(dim_z, dim_u, low_rank=True):
    A = torch.nn.Parameter(2. * (torch.randn(dim_z, dim_z) - .5))
    r = torch.nn.Parameter(2. * (torch.randn(dim_z) - .5))
    v = torch.nn.Parameter(2. * (torch.randn(dim_z) - .5))
    B = torch.nn.Parameter(2. * (torch.randn(dim_z, dim_u) - .5))
    o = torch.nn.Parameter(2. * (torch.randn(dim_z, 1) - .5))

    return LinearTransition(dim_z, dim_u, r, v, A, B, o, low_rank=low_rank)


def get_ball_pwa_transition(num_modes, dim_z, dim_u, low_rank=True):
    mode_classifier = nn.Linear(dim_z, num_modes)
    As = torch.nn.ParameterList()
    rs = torch.nn.ParameterList()
    vs = torch.nn.ParameterList()
    Bs = torch.nn.ParameterList()
    os = torch.nn.ParameterList()
    for mode in range(num_modes):
        As.append(torch.nn.Parameter(2. * (torch.randn(dim_z, dim_z) - .5)))
        rs.append(torch.nn.Parameter(2. * (torch.randn(dim_z) - .5)))
        vs.append(torch.nn.Parameter(2. * (torch.randn(dim_z) - .5)))
        Bs.append(torch.nn.Parameter(2. * (torch.randn(dim_z, dim_u) - .5)))
        os.append(torch.nn.Parameter(2. * (torch.randn(dim_z, 1) - .5)))

    return PWATransition(dim_z, dim_u, mode_classifier, rs, vs, As, Bs, os, low_rank=low_rank)
