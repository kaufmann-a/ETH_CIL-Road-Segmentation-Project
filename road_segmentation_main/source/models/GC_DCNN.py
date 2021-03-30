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
import torch.nn.functional as F
from torchsummary import summary

from source.models.basemodel import BaseModel


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
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding, dilation=1,
            ),

            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, dilation=1),  # TODO is padding = 1 ok?

            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, dilation=1)
        )

        # residual/identity connection
        # implemented residual/identity connection according to the resnet;
        #  see downsample in https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class conv2DBatchNormRelu(nn.Module):
    """
    Based on: https://github.com/shahabty/PSPNet-Pytorch/blob/f714cfe9b30eaef4dfb301cc261254e4a3c4b996/layers.py#L347
    """

    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True, dilation=1):
        super(conv2DBatchNormRelu, self).__init__()

        if dilation > 1:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=dilation)

        else:
            conv_mod = nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                 padding=padding, stride=stride, bias=bias, dilation=1)

        self.cbr_unit = nn.Sequential(conv_mod,
                                      nn.BatchNorm2d(int(n_filters)),
                                      nn.ReLU(inplace=True), )

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class PyramidPooling(nn.Module):
    """
    Pyramid Pooling Module - PPM
    Based on: https://github.com/shahabty/PSPNet-Pytorch/blob/f714cfe9b30eaef4dfb301cc261254e4a3c4b996/layers.py#L347
    """

    def __init__(self, in_channels, pool_sizes):
        super(PyramidPooling, self).__init__()

        self.paths = []
        for i in range(len(pool_sizes)):
            self.paths.append(conv2DBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=False))

        self.path_module_list = nn.ModuleList(self.paths)
        self.pool_sizes = pool_sizes

    def forward(self, x):
        output_slices = [x]
        h, w = x.shape[2:]

        for module, pool_size in zip(self.path_module_list, self.pool_sizes):
            out = F.avg_pool2d(x, int(h / pool_size), int(h / pool_size), 0)
            out = module(out)
            out = F.upsample(out, size=(h, w), mode='bilinear')
            output_slices.append(out)

        return torch.cat(output_slices, dim=1)


class GlobalContextDilatedCNN(BaseModel):
    """
    Global context dilated convolutional neural network
    """
    name = 'GCDCNN'

    def __init__(self, options, channel=3, filters=[64, 128, 256, 512]):
        super(GlobalContextDilatedCNN, self).__init__()

        # Encoder
        # level numbers according to paper
        # level 1
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
        )
        self.input_skip = nn.Sequential(  # TODO should we add BN?
            nn.Conv2d(channel, filters[0], kernel_size=1, stride=1, padding=0)
        )

        # Level 2 - RDB 1
        self.residual_dilated_block_1 = ResidualDilatedBlock(filters[0], filters[1], stride=2, padding=1)
        # Level 3 - RDB 2
        self.residual_dilated_block_2 = ResidualDilatedBlock(filters[1], filters[2], stride=2, padding=1)
        # Level 4 - RDB 3
        self.residual_dilated_block_3 = ResidualDilatedBlock(filters[2], filters[3], stride=2, padding=1)

        # Bridge
        # Level 5
        self.bridge = PyramidPooling(filters[3], [6, 3, 2, 1])

        # Decoder
        # Accordint to paper:
        #  Upsampling is done with the transposed convolution of pytorch using kernel size 2 and stride 2
        # Level 6
        self.upsample_1 = nn.ConvTranspose2d(filters[3] * 2, filters[2], kernel_size=2, stride=2)
        self.up_residual_block1 = ResidualBlock(filters[2] * 2, filters[2], stride=1, padding=1)

        # Level 7
        self.upsample_2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.up_residual_block2 = ResidualBlock(filters[1] * 2, filters[1], stride=1, padding=1)

        # Level 8
        self.upsample_3 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.up_residual_block3 = ResidualBlock(filters[0] * 2, filters[0], stride=1, padding=1)

        # Output
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], out_channels=1, kernel_size=1, stride=1),
            # nn.Sigmoid() # we use sigmoid later in the loss function
        )

    def forward(self, x):
        # Encoder
        # level numbers according to paper
        lv1 = self.input_layer(x) + self.input_skip(x)
        lv2 = self.residual_dilated_block_1(lv1)
        lv3 = self.residual_dilated_block_2(lv2)
        lv4 = self.residual_dilated_block_3(lv3)

        # Bridge
        lv5 = self.bridge(lv4)  # bride already concatenates level 4 with pyramid pooled output!

        # Decoder
        # Level 6
        x6 = self.upsample_1(lv5)
        x7 = torch.cat([x6, lv3], dim=1)
        x8 = self.up_residual_block1(x7)

        # Level 7
        x9 = self.upsample_2(x8)
        x10 = torch.cat([x9, lv2], dim=1)
        x11 = self.up_residual_block2(x10)

        # Level 8
        x12 = self.upsample_3(x11)
        x13 = torch.cat([x12, lv1], dim=1)
        x14 = self.up_residual_block3(x13)

        # Output
        output = self.output_layer(x14)

        return output


if __name__ == '__main__':
    """
    Test if the dimensions work out and print model
    """
    model = GlobalContextDilatedCNN(options=[])
    summary(model, input_size=(3, 256, 256), device="cpu")
