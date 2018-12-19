import torch
import torch.nn as nn

class MobileNet_CIFAR(nn.Module):
    def __init__(self):
        super(MobileNet_CIFAR, self).__init__()

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=1, bias=False),
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