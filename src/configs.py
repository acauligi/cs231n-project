"""
Configuration for the encoder, decoder, transition
for different tasks. Use load_config to find the proper
set of configuration.
"""
import torch
from torch import nn
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, dim_in, dim_out, conv_layers, pool_layers, ff_layers, conv_activation=None, ff_activation=None): 
        super(Encoder, self).__init__()
        self.m = enc

        self.dim_int = dim_in
        self.dim_out = dim_out

        self.conv_layers = conv_layers
        self.pool_layers = pool_layers
        
        self.ff_layers = ff_layers
        self.cnn_output_size = cnn_output_size
        
        self.conv_activation = conv_activation
        self.ff_activation = ff_activation

    def forward(self, x):
        for ii in range(0,len(self.conv_layers)):
            x = self.conv_layers[ii](x)
            if self.conv_activation:
                x = self.conv_activation(x)
            if self.pool_layers[ii]:
                x = self.pool_layers[ii](x)

        x = torch.flatten(x,start_dim=1)
        for ii in range(0,len(self.ff_layers)-1):
            x = self.ff_layers[ii](x)
            if self.ff_activation:
                x = self.ff_activation(x)
        return x.chunk(2, dim=1)

class Decoder(nn.Module):
    def __init__(self, dec, dim_in): 
        super(Decoder, self).__init__()
        self.m = dec
        self.dim_in = dim_in

    def forward(self, z):
        return self.m(z)


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


class BallEncoder(Encoder):
    def __init__(self, dim_in, dim_z, channels, ff_shape, conv_activation, ff_activation, stride, padding, pool): 
        conv_layers = []
        pool_layers = []
        W, H = input_size
        for ii in range(0,len(channels)-1):
            conv_layers.append(torch.nn.Conv2d(channels[ii], channels[ii+1], kernel[ii],
                stride=stride[ii], padding=padding[ii]))
            W = int(1+(W - kernel[ii] +2*padding[ii])/stride[ii])
            H = int(1+(H - kernel[ii] +2*padding[ii])/stride[ii])
            if pool[ii]:
                if W % pool[ii] != 0 or H % pool[ii] != 0:
                    raise ValueError('trying to pool by non-factor')
                W, H = W/pool[ii], H/pool[ii]
                pool_layers.append(torch.nn.MaxPool2d(pool[ii]))
            else:
                pool_layers.append(None)

        cnn_output_size = W*H*channels[-1]+num_features

        shape = np.concatenate(([cnn_output_size], ff_shape))
        for ii in range(0,len(shape)-1):
            self.ff_layers.append(torch.nn.Linear(shape[ii],shape[ii+1]))
        self.ff_layers.append(torch.nn.Linear(shape[ii], 2*dim_z))  # mean, diag of log(variance)

        conv_layers = torch.nn.ModuleList(conv_layers)
        ff_layers = torch.nn.ModuleList(ff_layers)
        if any(pool): 
            pool_layers = torch.nn.ModuleList(pool_layers)

        super(BallEncoder, self).__init__(m, dim_in, dim_z, conv_layers, pool_layers, ff_layers, conv_activation=conv_activation, ff_activation=ff_activation)

class BallDecoder(Decoder):
    def __init__(self, dim_in, dim_z, channels, ff_shape, conv_activation, stride, padding, pool, ff_activation): 
        out_channel = 16
        m = nn.Sequential(
          nn.ConvTranspose2d(in_channel, out_channel, 3, stride=2),
          nn.ReLU(),
          nn.ConvTranspose2D(out_channel, 
        )

        super(BallDecoder, self).__init__(m, dim_in) 


class BallTransition(Transition):
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

from e2c import NormalDistribution

__all__ = ['load_config']
