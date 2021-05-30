"""
Unet model
"""


__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from source.models.basemodel import BaseModel

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super(DoubleConv, self).__init__()
        padding = ((config.filtersize-1)*config.dilation+1)//2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=config.filtersize,
                      stride=config.stride, padding=padding, dilation=config.dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=config.filtersize,
                      stride=config.stride, padding=padding, dilation=config.dilation,  bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(BaseModel):
    name = 'unet'


    def __init__(self, config):
        super(UNET, self).__init__()
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.features = config.features

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=config.pool_kernelsize, stride=config.pool_stride, padding=config.pool_padding)

        # Down part of UNET
        for feature in self.features:
            self.downs.append(DoubleConv(self.in_channels, feature, config))
            self.in_channels = feature

        # Up part of UNET
        for feature in reversed(self.features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=config.pool_kernelsize, stride=config.pool_stride)
            )
            self.ups.append(DoubleConv(feature * 2, feature, config))

        self.bottleneck = DoubleConv(self.features[-1], self.features[-1] * 2, config)

        # Choose output kernel size
        if not config.use_submission_masks:
            out_kernel_size = 1
            out_stride = 1
        else:
            out_kernel_size = 16
            out_stride = 16

        self.final_conv = nn.Conv2d(self.features[0], self.out_channels, kernel_size=out_kernel_size, stride=out_stride)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2] #// = integer division

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:]) #skip batchsize and channels

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((3, 1, 160, 160))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()