#!/usr/bin/env python3
# coding: utf8

""""
GlobalContextDilatedCNN (GC-DCNN)

Implementation based on:
 Paper: Global context based automatic road segmentation via dilated convolutional neural network
        by Meng Lan, Yipeng Zhang, Lefei Zhang, Bo Du
 Link: https://www.sciencedirect.com/science/article/abs/pii/S0020025520304862

"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import torch
import torch.nn as nn

from source.models.basemodel import BaseModel
from source.models.modules import ResidualConv, Upsample


class ResidualDilatedBlock(nn.Module):
    """
    ResidualDilatedBlock - RDB
    """

    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualDilatedBlock, self).__init__()

        # TODO Should we set bias=False in the Conv2d layers, since the BN layer after it will "get rid of it":
        #  See:
        #  https://discuss.pytorch.org/t/any-purpose-to-set-bias-false-in-densenet-torchvision/22067,
        #  https://arxiv.org/pdf/1502.03167.pdf page 5,
        #  Why ResNet has bias=False: https://github.com/KaimingHe/deep-residual-networks/issues/10#issuecomment-194037195

        # dilated convolution
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding, dilation=1
            ),

            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=2, dilation=2),  # TODO is padding = dilation ok?

            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=2, dilation=2)
        )

        # residual/identity connection
        # implemented residual/identity connection according to the resnet;
        #  see downsample in https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class ResidualBlock(nn.Module):
    """
    ResidualDilatedBlock - RB
    """

    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualBlock, self).__init__()

        # dilated convolution
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding, dilation=1
            ),

            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, dilation=1),  # TODO is padding = dilation ok?

            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, dilation=1)
        )

        # residual/identity connection
        # implemented residual/identity connection according to the resnet;
        #  see downsample in https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, padding=0),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class PyramidPooling(nn.Module):
    """
    Pyramid Pooling Module - PPM
    """

    def __init__(self, input_dim, output_dim, stride, padding):
        super(PyramidPooling, self).__init__()
        # TODO implement

    def forward(self, x):
        # TODO implement
        pass


class GlobalContextDilatedCNN(BaseModel):
    """
    Global context dilated convolutional neural network
    """
    name = 'GC-DCNN'

    def __init__(self, options, channel=3, filters=[64, 128, 256, 512, 1024]):
        super(GlobalContextDilatedCNN, self).__init__()

        # Encoder
        # level 1
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
        )
        self.input_skip = nn.Sequential(  # TODO no BN?
            nn.Conv2d(channel, filters[0], kernel_size=1, stride=1, padding=0)
        )

        # Level 2 - RDB 1
        self.residual_dilated_block_1 = ResidualDilatedBlock(filters[0], filters[1], stride=2, padding=1)
        # Level 3 - RDB 2
        self.residual_dilated_block_2 = ResidualDilatedBlock(filters[1], filters[2], stride=2, padding=1)
        # Level 4 - RDB 3
        self.residual_dilated_block_2 = ResidualDilatedBlock(filters[2], filters[3], stride=2, padding=1)

        # Bridge
        # Level 5
        # TODO add pyramid pooling module
        # self.bridge = PyramidPooling(filters[3], filters[4])

        # Decoder
        # Level 6
        self.upsample_1 = Upsample(filters[4], filters[2], kernel=2, stride=2)
        self.up_residual_block1 = ResidualBlock(filters[3] + filters[2], filters[2], stride=1, padding=1)

        # Level 7
        self.upsample_2 = Upsample(filters[2], filters[2], kernel=2, stride=2)
        self.up_residual_block2 = ResidualBlock(filters[2] + filters[1], filters[1], stride=1, padding=1)

        # Level 8
        self.upsample_3 = Upsample(filters[1], filters[1], kernel=2, stride=2)
        self.up_residual_block3 = ResidualBlock(filters[1] + filters[0], filters[0], stride=1, padding=1)

        # Output
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], out_channels=1, kernel_size=1, stride=1),
            # nn.Sigmoid() # we use sigmoid later in the loss function
        )

    def forward(self, x):
        # Encoder
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_dilated_block_1(x1)
        x3 = self.residual_dilated_block_2(x2)
        x4 = self.residual_dilated_block_3(x3)

        # Bridge
        x5 = self.bridge(x3)

        # Decoder
        # TODO

        return None
