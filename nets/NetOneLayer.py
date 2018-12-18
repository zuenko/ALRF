import torch
import torch.nn as nn
import torch.nn.functional as F

class Net_one_layer(nn.Module):
    def __init__(self, n_hidden=2**8, d=8, K=2):
        # n_hidden - number of hidden units
        # d - rank of decompositions
        # K - number of pairs (U(k), V(k))
        super(Net_one_layer, self).__init__()
        self.n_hidden = n_hidden
        self.d = d
        self.K = K
        
        self.Vs1 = [nn.Parameter(torch.randn(784, d, requires_grad=True)) for k in range(self.K)]
        self.Us1 = [nn.Parameter(torch.randn(d, n_hidden, requires_grad=True)) for k in range(self.K)]
        for k in range(self.K):
            self.register_parameter('U{}'.format(k), self.Us1[k])
            self.register_parameter('V{}'.format(k), self.Vs1[k])
        
        self.W_pi = nn.Parameter(torch.randn(49, K), requires_grad=True)
        self.W2 = nn.Parameter(torch.randn(n_hidden, 10, requires_grad=True))
        
    def forward(self, x):
        # x has shape (28,28)
        x_pooled = F.max_pool2d(x, 4) # (28, 28) -> (7,7)
        x_pooled = x_pooled.view(-1, 7*7)
        pi = F.sigmoid(x_pooled.mm(self.W_pi))
        pi = pi.view((*pi.size(), 1)) # (n_samples, K, 1)
        
        x = x.view(x.size()[0], -1)
        
        # the next three line are magic
        Wx = [(x.mm(self.Vs1[k])).mm(self.Us1[k]) for k in range(self.K)]  
        Wx = torch.stack(Wx, dim=1)                        # (n_samples, K, n_hidden)
        Wx = (pi * Wx).sum(dim=1) # (n_samples, n_hidden)

        x = F.relu(Wx)
        y_pred = F.softmax(x.mm(self.W2), dim=1)
        return y_pred