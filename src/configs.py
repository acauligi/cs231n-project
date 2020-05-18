"""
Configuration for the encoder, decoder, transition
for different tasks. Use load_config to find the proper
set of configuration.
"""
import pdb
import torch
from torch import nn
from torch import nn,distributions
from torch.autograd import Variable
import numpy as np
import copy

class Encoder(nn.Module):
    def __init__(self, dim_in, dim_z, channels, ff_shape, kernel, stride, padding, pool, conv_activation=None, ff_activation=None): 
        super(Encoder, self).__init__()
        conv_layers = []
        conv_batch_layers = []
        ff_batch_layers = []
        pool_layers = []
        ff_layers = []

        C, W, H = dim_in
        for ii in range(0,len(channels)-1):
            conv_layers.append(torch.nn.Conv2d(channels[ii], channels[ii+1], kernel[ii],
                stride=stride[ii], padding=padding[ii]))
            conv_batch_layers.append(torch.nn.BatchNorm2d(channels[ii+1]))

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
          ff_batch_layers.append(torch.nn.BatchNorm1d(ff_shape[ii+1]))
        ff_layers.append(torch.nn.Linear(ff_shape[ii], 2*dim_z))  # mean, diag of log(variance)
        ff_batch_layers.append(torch.nn.BatchNorm1d(2*dim_z))

        self.dim_in = dim_in
        self.dim_out = dim_z 

        self.conv_layers = torch.nn.ModuleList(conv_layers)
        self.conv_batch_layers = torch.nn.ModuleList(conv_batch_layers)
        if any(pool): 
            self.pool_layers = torch.nn.ModuleList(pool_layers)
        else:
            self.pool_layers = pool_layers 
        
        self.ff_layers = torch.nn.ModuleList(ff_layers)
        self.ff_batch_layers = torch.nn.ModuleList(ff_batch_layers)

        self.conv_activation = conv_activation
        self.ff_activation = ff_activation

    def forward(self, x):
        # First compute convolutional pass
        for ii in range(0,len(self.conv_layers)):
            x = self.conv_layers[ii](x)
            if self.conv_activation:
                x = self.conv_activation(x)
            x = self.conv_batch_layers[ii](x)
            if self.pool_layers[ii]:
                x = self.pool_layers[ii](x)
 
        # Flatten output and compress
        x = x.view(x.shape[0], -1)
        for ii in range(0,len(self.ff_layers)):
            x = self.ff_layers[ii](x)
            # x = self.ff_batch_layers[ii](x)
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
        conv_batch_layers = []

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
            conv_batch_layers.append(torch.nn.BatchNorm2d(channels[ii+1]))

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.ff_layers = torch.nn.ModuleList(ff_layers)
        self.conv_layers = torch.nn.ModuleList(conv_layers)
        self.conv_batch_layers = torch.nn.ModuleList(conv_batch_layers)
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
            # x = self.conv_batch_layers[ii](x)
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

    def forward(self, hh, Q, uu):
        batch_size = hh.size()[0]
        vv, rr = self.trans(hh).chunk(2, dim=1)
        v1 = vv.unsqueeze(2)
        rT = rr.unsqueeze(1)
        I = Variable(torch.eye(self.dim_z).repeat(batch_size, 1, 1))
        if rT.data.is_cuda:
            # I.data.cuda()
            I = I.to(rT.device)
        A = I.add(v1.bmm(rT))

        if self.dim_u is not 0:
          B = self.fc_B(hh).view(-1, self.dim_z, self.dim_u)
        o = self.fc_o(hh)

        # need to compute the parameters for distributions
        # as well as for the samples
        if self.dim_u is not 0:
          uu = uu.unsqueeze(2)

        if self.dim_u is not 0:
          d = A.bmm(Q.mu.unsqueeze(2)).add(B.bmm(uu)).add(o).squeeze(2)
          sample = A.bmm(hh.unsqueeze(2)).add(B.bmm(uu)).add(o).squeeze(2)
        else:
          d = A.bmm(Q.mu.unsqueeze(2)).add(o.unsqueeze(2)).squeeze(2)
          sample = A.bmm(hh.unsqueeze(2)).add(o.unsqueeze(2)).squeeze(2)

        # z_cov = Q.covariance_matrix
        # z_next_cov = A.bmm(z_cov).bmm(A.transpose(1,2))

        return sample, NormalDistribution(d, Q.sigma, Q.logsigma, v=vv, r=rr)
        # return sample, distributions.MultivariateNormal(d,z_next_cov)


class BallEncoder(Encoder):
    def __init__(self, dim_in, dim_z): 
        channels_enc = [3, 32, 64]
        ff_shape = [512, 512]

        conv_activation = torch.nn.ReLU()
        ff_activation = torch.nn.ReLU()

        n_channels = len(channels_enc) - 1
        kernel_enc = [2,3]
        stride= [2] * n_channels
        padding= [2] * n_channels
        pool = [2, 2] * n_channels

        super(BallEncoder, self).__init__(dim_in, dim_z, channels_enc, ff_shape, kernel_enc, stride, padding, pool, conv_activation=conv_activation, ff_activation=ff_activation)

class BallDecoder(Decoder):
    def __init__(self, dim_in, dim_out): 
        channels_dec = [64, 32, dim_out[0]]
        ff_shape = [512, 512]

        conv_activation = torch.nn.ReLU()
        ff_activation = torch.nn.ReLU()

        n_channels = len(channels_dec) - 1
        kernel_dec = [2, 2]
        stride= [2] * n_channels
        padding= [2] * n_channels
        pool = [2, 2] * n_channels

        super(BallDecoder, self).__init__(dim_in, dim_out, channels_dec, ff_shape, kernel_dec, stride, padding, ff_activation=ff_activation, conv_activation=conv_activation)

class BallTransition(Transition):
    def __init__(self, dim_z, dim_u):
        trans = nn.Sequential(
            nn.Linear(dim_z, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, dim_z*2)
        )
        super(BallTransition, self).__init__(trans, dim_z, dim_u)


_CONFIG_MAP = {
    'ball': (BallEncoder, BallTransition, BallDecoder),
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
