import torch
import torch.nn as nn
import torch.nn.functional as F

class NetOneLayer(nn.Module):
    def __init__(self, n_hidden=2**8):
        # n_hidden - number of hidden units
        super(NetOneLayer, self).__init__()
        self.n_hidden = n_hidden
        
        self.W1 = nn.Parameter(torch.randn(784, self.n_hidden, requires_grad=True))
        self.W2 = nn.Parameter(torch.randn(self.n_hidden, 10, requires_grad=True))
        
    def forward(self, x):
        # x has shape (n_samples, 1, 28, 28)
        x = x.view(x.size()[0], -1)

        x = torch.sigmoid(x.mm(self.W1))
        y_pred = F.softmax(x.mm(self.W2), dim=1)
        return y_pred