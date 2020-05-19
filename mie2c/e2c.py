from mie2c.losses import binary_crossentropy

import torch
from torch import nn
from torch.autograd import Variable
from torch import nn, distributions
import numpy as np
import copy
import pdb

class Encoder(nn.Module):
    def __init__(self, dim_in, dim_z, channels, ff_shape, kernel, stride, padding, pool, conv_activation=None, ff_activation=None): 
        super().__init__()

        conv_layers = []
        batch_norm_layers = []
        pool_layers = []
        ff_layers = []

        C, W, H = dim_in
        for ii in range(0,len(channels)-1):
            conv_layers.append(torch.nn.Conv2d(channels[ii], channels[ii+1], kernel[ii],
                stride=stride[ii], padding=padding[ii]))
            batch_norm_layers.append(torch.nn.BatchNorm2d(channels[ii+1]))
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
        self.batch_norm_layers = torch.nn.ModuleList(batch_norm_layers)
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
            x = self.batch_norm_layers[ii](x)
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
        super().__init__()

        ff_shape = copy.copy(ff_shape)
        channels = copy.copy(channels)

        ff_layers = []
        conv_layers = []
        batch_norm_layers = []

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
            batch_norm_layers.append(torch.nn.BatchNorm2d(channels[ii+1]))

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.ff_layers = torch.nn.ModuleList(ff_layers)
        self.conv_layers = torch.nn.ModuleList(conv_layers)
        self.batch_norm_layers = torch.nn.ModuleList(batch_norm_layers)
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
            # x = self.batch_norm_layers[ii](x)
            if self.conv_activation:
                x = self.conv_activation(x)
        return x


class Transition(nn.Module):
    def __init__(self, trans, dim_z, dim_u):
        super().__init__()

        self.trans = trans
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.fc_B = nn.Linear(dim_z, dim_z * dim_u)
        self.fc_o = nn.Linear(dim_z, dim_z)

    def forward(self, h, Q, u):
        batch_size = h.size()[0]

        # computes the new basis vector for the embedded dynamics
        v, r = self.trans(h).chunk(2, dim=1)
        v1 = v.unsqueeze(2)
        rT = r.unsqueeze(1)
        I = Variable(torch.eye(self.dim_z).repeat(batch_size, 1, 1))
        if rT.data.is_cuda:
            I = I.to(rT.device)
        # A is batch_size X z_size X z_size
        A = I.add(v1.bmm(rT))
        # B is batch_size X z_size X input_size
        if self.dim_u is not 0:
          B = self.fc_B(h).view(-1, self.dim_z, self.dim_u)
          u = u.unsqueeze(2)
        # o (constant terms) is batch_size X z_size
        o = self.fc_o(h).unsqueeze(2)
        # need to compute the parameters for distributions
        # as well as for the samples

        if self.dim_u is not 0:
          d = A.bmm(Q.mean.float().unsqueeze(2)).add(B.bmm(u)).add(o).squeeze(2)
          sample = A.bmm(h.unsqueeze(2)).add(B.bmm(u)).add(o).squeeze(2)
        else:
          d = A.bmm(Q.mean.float().unsqueeze(2)).add(o).squeeze(2)
          sample = A.bmm(h.unsqueeze(2)).add(o).squeeze(2)

        Qz_next_cov = A.double().bmm(Q.covariance_matrix.double()).bmm(A.double().transpose(1,2))
        return sample, distributions.MultivariateNormal(d.double(), Qz_next_cov)


class LinearTransition(nn.Module):
    # def __init__(self, A, B, o):
    #     super().__init__()
    #     self.A = A
    #     self.B = B
    #     self.o = o
    def __init__(self, dim_z, dim_u, r, v, B, o):
        super().__init__()
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.r = r
        self.v = v
        self.B = B
        self.o = o

    def forward(self, h, Q, u):
        # A = self.A
        r = self.r
        v = self.v
        A = torch.eye(r.shape[0]) + r.unsqueeze(1) @ v.unsqueeze(0)

        B = self.B
        o = self.o

        if self.dim_u is not 0:
          d = Q.mean.float() @ A.T + u @ B.T + o.T
          sample = h @ A.T + u @ B.T + o.T
        else:
          d = Q.mean.float() @ A.T + o.T
          sample = h @ A.T + o.T
        Qz_next_cov = A.double() @ Q.covariance_matrix @ A.double().T
        return sample, distributions.MultivariateNormal(d.double(), Qz_next_cov)

class PWATransition(nn.Module):
    # def __init__(self, mode_classifier, As, Bs, os):
    #     super().__init__()
    #     self.mode_classifier = mode_classifier
    #     self.As = As
    #     self.Bs = Bs
    #     self.os = os
    def __init__(self, dim_z, dim_u, mode_classifier, rs, vs, Bs, os):
        super().__init__()
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.mode_classifier = mode_classifier
        self.rs = rs
        self.vs = vs
        self.Bs = Bs
        self.os = os

    def forward(self, h, Q, uu):
        # TODO: vectorize this
        d = torch.zeros_like(h)
        sample = torch.zeros(h.shape)
        alpha = torch.nn.functional.softmax(self.mode_classifier(h), dim=1)
        Qz_next_cov = torch.zeros_like(Q.covariance_matrix)
        for mode in range(alpha.shape[1]):
            # A = self.As[mode]
            r = self.rs[mode]
            v = self.vs[mode]
            A = torch.eye(r.shape[0]) + r.unsqueeze(1) @ v.unsqueeze(0)

            B = self.Bs[mode]
            o = self.os[mode]

            if self.dim_u is not 0:
              d += (Q.mean.float() @ A.T + uu @ B.T + o.T) * alpha[:, mode].unsqueeze(1)
              sample += (h @ A.T + uu @ B.T + o.T) * alpha[:, mode].unsqueeze(1)
            else:
              d += (Q.mean.float() @ A.T + o.T) * alpha[:, mode].unsqueeze(1)
              sample += (h @ A.T + o.T) * alpha[:, mode].unsqueeze(1)
            A_local = alpha[:,mode].double().view(-1,1,1) * A.double()
            Qz_next_cov += A_local @ Q.covariance_matrix.double() @ A_local.transpose(1,2)
        return sample, distributions.MultivariateNormal(d.double(), Qz_next_cov)

    def predict(self, h, Q, u):
        # TODO: vectorize this
        d = torch.zeros_like(h)
        sample = torch.zeros(h.shape)
        alpha = torch.nn.functional.softmax(self.mode_classifier(h), dim=1)
        alpha = (alpha == alpha.max(dim=1)[0].unsqueeze(1)).float()
        Qz_next_cov = torch.zeros_like(Q.covariance_matrix)
        for mode in range(alpha.shape[1]):
            # A = self.As[mode]
            r = self.rs[mode]
            v = self.vs[mode]
            A = torch.eye(r.shape[0]) + r.unsqueeze(1) @ v.unsqueeze(0)

            B = self.Bs[mode]
            o = self.os[mode]

            if self.dim_u is not 0:
              d += (Q.mean.float() @ A.T + u @ B.T + o.T) * alpha[:, mode].unsqueeze(1)
              sample += (h @ A.T + u @ B.T + o.T) * alpha[:, mode].unsqueeze(1)
            else:
              d += (Q.mean.float() @ A.T + o.T) * alpha[:, mode].unsqueeze(1)
              sample += (h @ A.T + o.T) * alpha[:, mode].unsqueeze(1)
            A_local = alpha[:,mode].double().view(-1,1,1) * A.double()
            Qz_next_cov += A_local @ Q.covariance_matrix.double() @ A_local.transpose(1,2)
        return sample, distributions.MultivariateNormal(d.double(), Qz_next_cov)


class E2C(nn.Module):
    def __init__(self, enc, trans, dec):
        """Constructor for BnBCNN.
            dim_in: tuple of image size (C,W,H)
            dim_u: dimension of control vector in latent space
            channels: vector of length N+1 specifying # of channels for each convolutional layer,
            ff_shape: vector specifying shape of feedforward network. ff_shape[0] should be 
                the size of the first hidden layer; constructor does the math to determine ff input size.
                where N is number of conv layers. channels[0] should be the size of the input image.
            conv_activation: nonlinear activation to be used after each conv layer
            ff_activation: nonlinear activation to be used after each ff layer
            kernel: vector (or scalar) of kernel sizes for each conv layer. if scalar, each layer
                uses the same kernel.
            stride: list of strides for CNN layers.
            padding: list of paddings for CNN layers.
            pool: pooling to be added after each layer. if None, no pooling. if scalar, same pooling for each layer.
        """
        super().__init__()
        self.encoder = enc
        self.decoder = dec 
        self.trans = trans

        self.prior = distributions.Normal(0, 1)

    def encode(self, x):
        mean, logvar = self.encoder(x)
        return mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def transition(self, z, Qz, u):
        return self.trans(z, Qz, u)

    def reparam(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        self.z_mean = mean
        self.z_sigma = std
        eps = self.prior.sample()
        if std.data.is_cuda:
          eps = eps.to(std.device)

        cov = []
        for _ in std:
            cov.append(torch.diag(_))
        cov = torch.stack(cov,dim=0)
        return eps.mul(std).add_(mean), distributions.MultivariateNormal(mean.double(), cov.double())

    def forward(self, x, action, x_next):
        mean, logvar = self.encode(x)
        mean_next, logvar_next = self.encode(x_next)

        z, self.Qz = self.reparam(mean, logvar)
        z_next, self.Qz_next = self.reparam(mean_next, logvar_next)

        self.x_dec = self.decode(z)
        self.x_next_dec = self.decode(z_next)

        self.z_next_pred, self.Qz_next_pred = self.transition(z, self.Qz, action)
        self.x_next_pred_dec = self.decode(self.z_next_pred)

    def latent_embeddings(self, x):
        return self.encode(x)[0]

    def predict(self, X, U):
        mean, logvar = self.encoder.eval()(X)
        z, Qz = self.reparam(mean, logvar)
        if isinstance(self.trans, PWATransition):
            z_next_pred, Qz_next_pred = self.trans.predict(z, Qz, U)
        else:  
            z_next_pred, Qz_next_pred = self.trans.eval()(z, Qz, U)
        x_next_dec = self.decoder.eval()(z_next_pred)
        return x_next_dec

def compute_loss(x_dec, x_next_dec, x_next_pred_dec,
                 x, x_next,
                 Qz, Qz_next, Qz_next_pred):

    x_reconst_loss = (x_dec - x).pow(2).sum(dim=[1,2,3])
    x_next_reconst_loss = (x_next_pred_dec - x_next).pow(2).sum(dim=[1,2,3])

    prior = distributions.MultivariateNormal(torch.zeros_like(Qz.mean[0]).double(),torch.diag(torch.ones_like(Qz.mean[0])).double())
    KLD = distributions.kl_divergence(Qz,prior) + distributions.kl_divergence(Qz_next,prior)

    # ELBO
    bound_loss = x_reconst_loss.add(x_next_reconst_loss).double().add(KLD)
    trans_loss = distributions.kl_divergence(Qz_next_pred, Qz_next) # .add(x_next_pre_reconst_loss)

    return bound_loss.mean()/2, trans_loss.mean()
