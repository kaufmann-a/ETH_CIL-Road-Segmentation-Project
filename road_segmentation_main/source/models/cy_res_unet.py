"""
 Adapted and modified from: https://github.com/rishikksh20/ResUnet to get the CyResUnet.
"""

import torch
import torch.nn as nn

from source.models.cyconvlayer import CyConv2d
from source.models.basemodel import BaseModel


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            CyConv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            CyConv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            CyConv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(  # TODO
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class CyResUnet(BaseModel):
    name = 'CyResUnet'

    def __init__(self, config, channel=3, filters=[64, 128, 256, 512]):
        super(CyResUnet, self).__init__()

        # Encoding
        # level 1
        self.input_layer = nn.Sequential(
            CyConv2d(channel, filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            CyConv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
        )
        self.input_skip = nn.Sequential(
            CyConv2d(channel, filters[0], kernel_size=3, stride=1, padding=1)
        )

        # Level 2
        self.residual_conv_1 = ResidualConv(filters[0], filters[1], stride=2, padding=1)
        # Level 3
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], stride=2, padding=1)

        # Bridge
        # Level 4
        self.bridge = ResidualConv(filters[2], filters[3], stride=2, padding=1)

        # Decoding
        # Level 5
        self.upsample_1 = Upsample(filters[3], filters[3], kernel=2, stride=2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], stride=1, padding=1)

        # Level 6
        self.upsample_2 = Upsample(filters[2], filters[2], kernel=2, stride=2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], stride=1, padding=1)

        # Level 7
        self.upsample_3 = Upsample(filters[1], filters[1], kernel=2, stride=2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], stride=1, padding=1)

        # Choose output kernel size
        if not config.use_submission_masks:
            out_kernel_size = 1
            out_stride = 1
        else:
            out_kernel_size = 16
            out_stride = 16

        self.output_layer = nn.Sequential(
            CyConv2d(filters[0], out_channels=1, kernel_size=out_kernel_size, stride=out_stride),
            # nn.Sigmoid() # we do use sigmoid later in the loss function
        )

    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)
        del x3, x4

        x6 = self.up_residual_conv1(x5)
        del x5

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)
        del x2, x6

        x8 = self.up_residual_conv2(x7)
        del x7

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)
        del x1, x8

        x9 = self.up_residual_conv3(x9)

        output = self.output_layer(x9)

        return output
