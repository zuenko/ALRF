import torch
import torch.nn as nn
import torch.nn.functional as F

from .LowRankLayer import *

class NetMLPLowRank(nn.Sequential):
    ### Multi-layer perceptron with adaptive low-rank hidden layers
    def __init__(self, input_size=28*28, output_size=10, n_layers=1, n_hidden=[2**8], d=[2], K=[2], pi_size=[28]):
        assert n_layers == len(n_hidden) == len(d) == len(K) == len(pi_size)
        
        layers = () 
        input_sizes = [input_size] + n_hidden[:-1]
        output_sizes = n_hidden
        for i in range(n_layers):
            layers = (
                *layers,
                LowRankLayer(input_sizes[i], output_sizes[i], d=d[i], K=K[i], pi_size=pi_size[i]),
                nn.Sigmoid(),
            )
        layers = (*layers, FlattenBatch(), nn.Linear(in_features=n_hidden[-1], out_features=output_size))
        
        super(NetMLPLowRank, self).__init__(*layers)