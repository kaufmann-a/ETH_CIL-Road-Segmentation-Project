"""
 Based on: https://github.com/rishikksh20/ResUnet,
   https://github.com/nikhilroxtomar/Deep-Residual-Unet/blob/master/Deep%20Residual%20UNet.ipynb,
   https://github.com/Kaido0/Brain-Tissue-Segment-Keras/blob/4ce27857b0bec8109c03db8263a9b2f41e3a6a55/net/res_unet.py
"""

import torch
import torch.nn as nn

from source.models.basemodel import BaseModel
from source.models.modules import ResidualConv, Upsample


class ResUnet(BaseModel):
    """
    Based on https://arxiv.org/pdf/1711.10684.pdf

    """
    name = 'ResUnet'

    def __init__(self, config, channel=3, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        # Encoding
        # level 1
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(filters[0])
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
            nn.Conv2d(filters[0], out_channels=1, kernel_size=out_kernel_size, stride=out_stride),
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

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output
