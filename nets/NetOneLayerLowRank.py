import torch
import torch.nn as nn
import torch.nn.functional as F

class NetOneLayerLowRank(nn.Module):
    def __init__(self, n_hidden=2**8, d=8, K=2):
        # n_hidden - number of hidden units
        # d - rank of decompositions
        # K - number of pairs (U(k), V(k))
        super(NetOneLayerLowRank, self).__init__()
        self.n_hidden = n_hidden
        self.d = d
        self.K = K
        
        self.Vs1 = [nn.Parameter(torch.randn(784, self.d, requires_grad=True)) for k in range(self.K)]
        self.Us1 = [nn.Parameter(torch.randn(self.d, self.n_hidden, requires_grad=True)) for k in range(self.K)]
        for k in range(self.K):
            self.register_parameter('U{}'.format(k), self.Us1[k])
            self.register_parameter('V{}'.format(k), self.Vs1[k])
        
#         self.pool = nn.AvgPool1d(28)
#         self.W_pi = nn.Parameter(torch.randn(49, self.K), requires_grad=True)
        self.W_pi = nn.Parameter(torch.randn(28, self.K))
        self.W2 = nn.Parameter(torch.randn(self.n_hidden, 10, requires_grad=True))
        
    def forward(self, x):
        # x has shape (n_samples, 1, 28, 28)
        x = x.view(x.shape[0], 1, 28*28)
        x_pooled = F.avg_pool1d(x, 28).view(x.shape[0], 28)
        pi = torch.sigmoid(x_pooled.mm(self.W_pi))
        pi = pi.view((*pi.size(), 1)) # (n_samples, K, 1)
        
#         # x has shape (n_samples, 1, 28, 28)
#         x_pooled = F.max_pool2d(x, 4) # (28, 28) -> (7,7)
#         x_pooled = x_pooled.view(-1, 7*7)
#         pi = torch.sigmoid(x_pooled.mm(self.W_pi))
#         pi = pi.view((*pi.size(), 1)) # (n_samples, K, 1)

        x = x.view(x.shape[0], -1)
        
        # the next three line are magic
        Wx = [(x.mm(self.Vs1[k])).mm(self.Us1[k]) for k in range(self.K)]  
        Wx = torch.stack(Wx, dim=1)                        # (n_samples, K, n_hidden)
        Wx = (pi * Wx).sum(dim=1) # (n_samples, n_hidden)

        x = torch.sigmoid(Wx)
        y_pred = F.softmax(x.mm(self.W2), dim=1)
        return y_pred