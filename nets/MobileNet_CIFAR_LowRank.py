import torch
import torch.nn as nn
from .LowRankLayer import *
from math import sqrt

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        assert len(x.shape) == 4
        x = x.view(x.shape[0], x.shape[1], -1) # flattens (w,h) -> w*h
        x = x.permute(0,2,1) # permutes 1'st and 2'nd dimension, i.e. move n_channels to the last axis
        return x
    

class Unflatten(nn.Module):
    def __init__(self):
        super(Unflatten, self).__init__()

    def forward(self, x):
        assert len(x.shape) == 3
        x = x.permute(0,2,1)
        w = h = int(sqrt(x.shape[2]))
        x = x.view(x.shape[0], x.shape[1], w, h)
        return x

    
class MobileNet_CIFAR_LowRank(nn.Module):
    def __init__(self):
        super(MobileNet_CIFAR_LowRank, self).__init__()
                
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                
                Flatten(),
                LowRankLayer(inp, oup, d=8, K=2, pi_size=8, adaptive=True),
                Unflatten(),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            conv_dw( 32,  32, 1),
            conv_dw( 32,  32, 1),
            conv_dw( 32,  64, 1),
            
            conv_dw(64, 64, 2),
            conv_dw(64, 64, 2),
            conv_dw(64, 64, 2),
            conv_dw(64, 128, 2),

            conv_dw(128, 128, 2),
            conv_dw(128, 128, 2),
            conv_dw(128, 128, 2),
            conv_dw(128, 256, 2),
            nn.AvgPool2d(8, ceil_mode=True, count_include_pad=True),
        )
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x