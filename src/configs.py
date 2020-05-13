"""
Configuration for the encoder, decoder, transition
for different tasks. Use load_config to find the proper
set of configuration.
"""
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import copy

class Encoder(nn.Module):
    def __init__(self, dim_in, dim_z, channels, ff_shape, kernel, stride, padding, pool, conv_activation=None, ff_activation=None): 
        super(Encoder, self).__init__()
        conv_layers = []
        pool_layers = []
        ff_layers = []

        C, W, H = dim_in
        for ii in range(0,len(channels)-1):
            conv_layers.append(torch.nn.Conv2d(channels[ii], channels[ii+1], kernel[ii],
                stride=stride[ii], padding=padding[ii]))

            # Keep track of output image size
            W = int(1+(W - kernel[ii] +2*padding[ii])/stride[ii])
            H = int(1+(H - kernel[ii] +2*padding[ii])/stride[ii])
            if pool[ii]:
                if W % pool[ii] != 0 or H % pool[ii] != 0:
                    raise ValueError('trying to pool by non-factor')
                W, H = int(W/pool[ii]), int(H/pool[ii])
                pool_layers.append(torch.nn.MaxPool2d(pool[ii]))
            else:
                pool_layers.append(None)

        self.cnn_output_size = W*H*channels[-1]

        ff_shape = np.concatenate(([self.cnn_output_size], ff_shape))
        for ii in range(0, len(ff_shape) - 1):
          ff_layers.append(torch.nn.Linear(ff_shape[ii], ff_shape[ii+1]))
        ff_layers.append(torch.nn.Linear(ff_shape[ii], 2*dim_z))  # mean, diag of log(variance)

        self.dim_in = dim_in
        self.dim_out = dim_z 

        self.conv_layers = torch.nn.ModuleList(conv_layers)
        if any(pool): 
            self.pool_layers = torch.nn.ModuleList(pool_layers)
        else:
            self.pool_layers = pool_layers 
        
        self.ff_layers = torch.nn.ModuleList(ff_layers)

        self.conv_activation = conv_activation
        self.ff_activation = ff_activation

    def forward(self, x):
        # First compute convolutional pass
        for ii in range(0,len(self.conv_layers)):
            x = self.conv_layers[ii](x)
            if self.conv_activation:
                x = self.conv_activation(x)
            if self.pool_layers[ii]:
                x = self.pool_layers[ii](x)
 
        # Flatten output and compress
        x = x.view(x.shape[0], -1)
        for ii in range(0,len(self.ff_layers)):
            x = self.ff_layers[ii](x)
            if self.ff_activation:
                x = self.ff_activation(x)
        return x.chunk(2, dim=1)

class Decoder(nn.Module):
    def __init__(self, dim_in, dim_out, channels, ff_shape, kernel, stride, padding, ff_activation=None, conv_activation=None):
        super(Decoder, self).__init__()

        ff_shape = copy.copy(ff_shape)
        channels = copy.copy(channels)

        ff_layers = []
        conv_layers = []

        # Work backwards to figure out what flattened
        # input dimension should be for conv input
        C, W, H = dim_out
        for ii in range(len(channels)-2, -1, -1):
          W = int(1 + (W - kernel[ii] +2*padding[ii])/stride[ii])
          H = int(1 + (H - kernel[ii] +2*padding[ii])/stride[ii])

        cnn_input_size = (channels[0],W,H)  # input image to conv portion set to 1 channel

        ff_shape.insert(0, dim_in)
        for ii in range(0, len(ff_shape)-1):
          ff_layers.append(torch.nn.Linear(ff_shape[ii], ff_shape[ii+1]))
        ff_layers.append(torch.nn.Linear(ff_shape[-1], np.prod(list(cnn_input_size))))

        for ii in range(0, len(channels)-1):
            conv_layers.append(torch.nn.ConvTranspose2d(channels[ii], channels[ii+1], kernel[ii],
                stride=stride[ii], padding=padding[ii]))

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.ff_layers = torch.nn.ModuleList(ff_layers)
        self.conv_layers = torch.nn.ModuleList(conv_layers)
        self.cnn_input_size = cnn_input_size

        self.ff_activation = ff_activation
        self.conv_activation = conv_activation

    def forward(self, x):
        # First compute feedforward passes
        for ii in range(0,len(self.ff_layers)):
            x = self.ff_layers[ii](x)
            if self.ff_activation:
                x = self.ff_activation(x)

        # Deconvolutional passes
        # TODO(acauligi): reshape vector into tensor
        N = x.shape[0]
        C, W, H = self.cnn_input_size
        x = x.view(N,C,W,H)
        for ii in range(0, len(self.conv_layers)):
            x = self.conv_layers[ii](x)
            if self.conv_activation:
                x = self.conv_activation(x)
        return x

class Transition(nn.Module):
    def __init__(self, trans, dim_z, dim_u):
        super(Transition, self).__init__()
        self.trans = trans
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.fc_B = nn.Linear(dim_z, dim_z * dim_u)
        self.fc_o = nn.Linear(dim_z, dim_z)

    def forward(self, h, Q, u):
        batch_size = h.size()[0]
        v, r = self.trans(h).chunk(2, dim=1)
        v1 = v.unsqueeze(2)
        rT = r.unsqueeze(1)
        I = Variable(torch.eye(self.dim_z).repeat(batch_size, 1, 1))
        if rT.data.is_cuda:
            I.dada.cuda()
        A = I.add(v1.bmm(rT))

        B = self.fc_B(h).view(-1, self.dim_z, self.dim_u)
        o = self.fc_o(h)

        # need to compute the parameters for distributions
        # as well as for the samples
        u = u.unsqueeze(2)

        d = A.bmm(Q.mu.unsqueeze(2)).add(B.bmm(u)).add(o).squeeze(2)
        sample = A.bmm(h.unsqueeze(2)).add(B.bmm(u)).add(o).squeeze(2)

        return sample, NormalDistribution(d, Q.sigma, Q.logsigma, v=v, r=r)


class CifarEncoder(Encoder):
    def __init__(self, dim_in, dim_z): 
        channels_enc = [3, 8, 8]
        ff_shape = [32, 32]

        conv_activation = torch.nn.ReLU()
        ff_activation = torch.nn.ReLU()

        n_channels = len(channels_enc) - 1
        kernel_enc = [2,3]
        stride= [2] * n_channels
        padding= [2] * n_channels
        pool = [2, 2] * n_channels

        super(CifarEncoder, self).__init__(dim_in, dim_z, channels_enc, ff_shape, kernel_enc, stride, padding, pool, conv_activation=conv_activation, ff_activation=ff_activation)

class CifarDecoder(Decoder):
    def __init__(self, dim_in, dim_out): 
        channels_dec = [8, 8, dim_out[0]]
        ff_shape = [32, 32]

        conv_activation = torch.nn.ReLU()
        ff_activation = torch.nn.ReLU()

        n_channels = len(channels_dec) - 1
        kernel_dec = [2, 2]
        stride= [2] * n_channels
        padding= [2] * n_channels
        pool = [2, 2] * n_channels

        super(CifarDecoder, self).__init__(dim_in, dim_out, channels_dec, ff_shape, kernel_dec, stride, padding, ff_activation=ff_activation, conv_activation=conv_activation)

class CifarTransition(Transition):
    def __init__(self, dim_z, dim_u):
        trans = nn.Sequential(
            nn.Linear(dim_z, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, dim_z*2)
        )
        super(CifarTransition, self).__init__(trans, dim_z, dim_u)


_CONFIG_MAP = {
    'cifar': (CifarEncoder, CifarTransition, CifarDecoder),
}


def load_config(name):
    """Load a particular configuration
    Returns:
    (encoder, transition, decoder) A tuple containing class constructors
    """
    if name not in _CONFIG_MAP.keys():
        raise ValueError("Unknown config: %s", name)
    return _CONFIG_MAP[name]

from .e2c import NormalDistribution

__all__ = ['load_config']
