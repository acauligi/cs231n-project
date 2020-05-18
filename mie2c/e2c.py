from mie2c.losses import binary_crossentropy

import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import copy


class NormalDistribution:
    """
    Wrapper class representing a multivariate normal distribution parameterized by
    N(mu,Cov). If cov. matrix is diagonal, Cov=(sigma).^2. Otherwise,
    Cov=A*(sigma).^2*A', where A = (I+v*r^T).
    """
    def __init__(self, mu, sigma, logsigma, *, v=None, r=None):
        self.mu = mu
        self.sigma = sigma
        self.logsigma = logsigma
        self.v = v
        self.r = r

    @property
    def cov(self):
        """This should only be called when NormalDistribution represents one sample"""
        if self.v is not None and self.r is not None:
            assert self.v.dim() == 1
            dim = self.v.dim()
            v = self.v.unsqueeze(1)  # D * 1 vector
            rt = self.r.unsqueeze(0)  # 1 * D vector
            A = torch.eye(dim) + v.mm(rt)
            return A.mm(torch.diag(self.sigma.pow(2)).mm(A.t()))
        else:
            return torch.diag(self.sigma.pow(2))


def KLDGaussian(Q, N, eps=1e-8):
    """KL Divergence between two Gaussians
        Assuming Q ~ N(mu0, A\sigma_0A') where A = I + vr^{T}
        and      N ~ N(mu1, \sigma_1)
    """
    sum = lambda x: torch.sum(x, dim=1)
    k = float(Q.mu.size()[1])  # dimension of distribution
    mu0, v, r, mu1 = Q.mu, Q.v, Q.r, N.mu
    s02, s12 = (Q.sigma).pow(2) + eps, (N.sigma).pow(2) + eps
    a = sum(s02 * (1. + 2. * v * r) / s12) + sum(v.pow(2) / s12) * sum(r.pow(2) * s02)  # trace term
    b = sum((mu1 - mu0).pow(2) / s12)  # difference-of-means term
    c = 2. * (sum(N.logsigma - Q.logsigma) - torch.log(1. + sum(v * r) + eps))  # ratio-of-determinants term.
    return 0.5 * (a + b - k + c)


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
            x = self.batch_norm_layers[ii](x)
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
            I.dada.cuda()
        # A is batch_size X z_size X z_size
        A = I.add(v1.bmm(rT))

        # B is batch_size X z_size X input_size
        B = self.fc_B(h).view(-1, self.dim_z, self.dim_u)
        
        # o (constant terms) is batch_size X z_size
        o = self.fc_o(h).unsqueeze(2)

        # need to compute the parameters for distributions
        # as well as for the samples
        u = u.unsqueeze(2)

        d = A.bmm(Q.mu.unsqueeze(2)).add(B.bmm(u)).add(o).squeeze(2)
        sample = A.bmm(h.unsqueeze(2)).add(B.bmm(u)).add(o).squeeze(2)

        return sample, NormalDistribution(d, Q.sigma, Q.logsigma, v=v, r=r)


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
        eps = torch.FloatTensor(std.size()).normal_()
        if std.data.is_cuda:
            eps.cuda()
        eps = Variable(eps)
        return eps.mul(std).add_(mean), NormalDistribution(mean, std, torch.log(std))

    def forward(self, x, action, x_next):
        mean, logvar = self.encode(x)
        mean_next, logvar_next = self.encode(x_next)

        z, self.Qz = self.reparam(mean, logvar)
        z_next, self.Qz_next = self.reparam(mean_next, logvar_next)

        # self.x_dec = self.decode(z)
        # self.x_next_dec = self.decode(z_next)

        self.x_dec = self.decode(mean)
        self.x_next_dec = self.decode(mean_next)

    #     self.z_next_pred, self.Qz_next_pred = self.transition(z, self.Qz, action)
    #     self.x_next_pred_dec = self.decode(self.z_next_pred)

    def latent_embeddings(self, x):
        return self.encode(x)[0]

    def predict(self, X, U):
        # mean, logvar = self.encode(X)
        # z, Qz = self.reparam(mean, logvar)
    #     z_next_pred, Qz_next_pred = self.transition(z, Qz, U)
        # return self.decode(z_next_pred)
        # return self.decode(z)
        # return self.decode(mean)
        mean, logvar = self.encoder.eval()(X)
        x_dec = self.decoder.eval()(mean)
        return x_dec

# def compute_loss(x_dec, x_next_pred_dec, x, x_next,
#                  Qz, Qz_next_pred,
#                  Qz_next, mse=False):
def compute_loss(x_dec, x_next_dec, x, x_next,
                 Qz, Qz_next, mse=False):
    # Reconstruction losses
    # if mse:
    #     x_reconst_loss = (x_dec - x).pow(2).sum(dim=[1,2,3])
    #     x_next_reconst_loss = (x_next_pred_dec - x_next).pow(2).sum(dim=[1,2,3])
    # else:
    #     x_reconst_loss = -binary_crossentropy(x, x_dec).sum(dim=1)
    #     x_next_reconst_loss = -binary_crossentropy(x_next, x_next_pred_dec).sum(dim=1)

    if mse:
        x_reconst_loss = (x_dec - x).pow(2).sum(dim=[1,2,3])
        x_next_reconst_loss = (x_next_dec - x_next).pow(2).sum(dim=[1,2,3])
    else:
        x_reconst_loss = -binary_crossentropy(x, x_dec).sum(dim=1)
        x_next_reconst_loss = -binary_crossentropy(x_next, x_next_dec).sum(dim=1)

    # logvar = Qz.logsigma.mul(2)
    # KLD_element = Qz.mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # KLD = torch.sum(KLD_element, dim=1).mul(-0.5)

    # ELBO
    # bound_loss = x_reconst_loss + x_next_reconst_loss + KLD

    # bound_loss = x_reconst_loss + x_next_reconst_loss

    bound_loss = x_reconst_loss

    # bound_loss = x_reconst_loss.add(x_next_reconst_loss).add(KLD.view(-1,1,1))

    # kl = KLDGaussian(Qz_next_pred, Qz_next)

    # kl = torch.zeros(bound_loss.shape[0])
    # for i in range(bound_loss.shape[0]):
        # m1 = torch.distributions.MultivariateNormal(Qz_next_pred.mu[i,:], torch.diag(Qz_next_pred.sigma[i,:]))
        # m2 = torch.distributions.MultivariateNormal(Qz_next.mu[i,:], torch.diag(Qz_next.sigma[i,:]))
        # kl[i] = torch.distributions.kl.kl_divergence(m1, m2)

    # if torch.any(torch.isnan(Qz_next_pred.mu)):
    #     print("1")
    # if torch.any(torch.isnan(Qz_next_pred.sigma)):
    #     print("2")
    # if torch.any(torch.isnan(Qz_next.mu)):
    #     print("3")
    # if torch.any(torch.isnan(Qz_next.sigma)):
    #     print("4")
    # if torch.any(torch.isnan(kl)):
    #     print("5")
    # if torch.any(torch.isnan(kl)):
    #     print("6")

    # return bound_loss.mean(), kl.mean()

    return bound_loss.mean()
