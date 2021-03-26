import torch


def binary_crossentropy(t, o, eps=1e-8):
    return t * torch.log(o + eps) + (1.0 - t) * torch.log(1.0 - o + eps)


def kl_bernoulli(p, q, eps=1e-8):
    # http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/
    kl = p * torch.log((p + eps) / (q + eps)) + \
         (1 - p) * torch.log((1 - p + eps) / (1 - q + eps))
    return kl.mean()


class SigmoidAnneal:
    def __init__(self, dtype, lo, up, center_step, steps_lo_to_up):
        """
        provides a sigmoid function that can be used to do weight scheduling
        for training
        @dtype torch data type
        @param lo float lower value for the sigmoid
        @param up float upper value for the sigmoid
        @param center_step int step where the sigmoid will be halfway up
        @param steps_lo_to_up in width of the sigmoid
        """
        self.dtype = dtype
        self.lo = lo
        self.up = up
        self.center_step = center_step
        self.steps_lo_to_up = steps_lo_to_up
        self.sigmoid = torch.nn.Sigmoid()

    def __call__(self, step):
        """
        @param step int step number
        @return value of the sigmoid at that point
        """
        return self.lo + (self.up - self.lo) * self.sigmoid(torch.tensor(
            float(step - self.center_step) / float(self.steps_lo_to_up),
            dtype=self.dtype))