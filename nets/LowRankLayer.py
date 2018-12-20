import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms


class FlattenBatch(nn.Module):
    def __init__(self):
        super(FlattenBatch, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


class LowRankLayer(nn.Module):
    def __init__(self, input_size, output_size=2 ** 8, d=8, K=2, pi_size=28, adaptive=True):
        """
        d: rank of decompositions
        K: number of pairs (U(k), V(k))
        pi_size: size of pi function hidden layer
        """
        super(LowRankLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.d = d
        self.K = K
        self.pi_size = pi_size
        self.adaptive = adaptive

        self.Vs1 = [nn.Parameter(torch.Tensor(self.input_size, self.d), requires_grad=True) for k in range(self.K)]
        self.Us1 = [nn.Parameter(torch.Tensor(self.d, self.output_size), requires_grad=True) for k in range(self.K)]
        for k in range(self.K):
            nn.init.kaiming_uniform_(self.Vs1[k], a=2)
            nn.init.kaiming_uniform_(self.Us1[k], a=2)
            self.register_parameter('U{}'.format(k), self.Us1[k])
            self.register_parameter('V{}'.format(k), self.Vs1[k])

        if self.adaptive:
            self.W_pi = nn.Parameter(torch.Tensor(self.pi_size, self.K), requires_grad=True)
            nn.init.kaiming_uniform_(self.W_pi, a=2)
        else:
            self.W_pi = None

    def forward(self, x):
        # x has shape (n_samples, channel_count, input_size)
        if len(x.shape) > 3:
            x = x.view(x.shape[0], x.shape[1], -1)
            
        if self.adaptive:
            pool_size = x.shape[2] // self.pi_size
            x_pooled = F.avg_pool1d(x, pool_size)  # (n_samples, channel_count, pi_size)
            pi = x_pooled.matmul(self.W_pi)  # (n_samples, channel_count, K)
        else:
            pi = torch.full((x.shape[0], x.shape[1], self.K), 1 / self.K)

        pi = torch.sigmoid(pi)

        Wx = [(x.matmul(self.Vs1[k])).matmul(self.Us1[k]) for k in range(self.K)]
        Wx = torch.stack(Wx, dim=2)  # (n_samples, channel_count, K, output_size)
        Wx = (pi.view(*pi.shape, 1) * Wx).sum(dim=2)  # (n_samples, channel_count, output_size)
        
        return Wx