#!/usr/bin/env python3
# coding: utf8

""""
GlobalContextDilatedCNN (GC-DCNN) + Attention

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
from source.models.gcdcnn import ResidualDilatedBlock
from source.models.modules import PPM, AttentionBlock, ASPP, ASPP_new, Upsample_


class GlobalContextDilatedCNNPlus(BaseModel):
    """
    Global context dilated convolutional neural network with attention
    """
    name = 'gcdcnn_plus'

    def __init__(self, config, channel=3, filters=[64, 128, 256, 512]):
        super(GlobalContextDilatedCNNPlus, self).__init__()

        # Encoder
        # level numbers according to paper
        # level 1
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=1, stride=1, padding=0)
            # nn.BatchNorm2d(filters[0]) # TODO add BN?
        )

        # Level 2 - RDB 1
        self.residual_dilated_block_1 = ResidualDilatedBlock(filters[0], filters[1], stride=2, dilation=2, padding=1)
        # Level 3 - RDB 2
        self.residual_dilated_block_2 = ResidualDilatedBlock(filters[1], filters[2], stride=2, dilation=2, padding=1)
        # Level 4 - RDB 3
        self.residual_dilated_block_3 = ResidualDilatedBlock(filters[2], filters[3], stride=2, dilation=2, padding=1)

        # TODO not clear where they use BN: during "upsampling" and in ppm

        # Bridge
        # Level 5
        self.bridge = ASPP_new(filters[3], filters[3], use_global_avg_pooling=False)

        # Decoder
        # Accordint to paper:
        #  "Upsampling is done with the transposed convolution of pytorch using kernel size 2 and stride 2"

        CONV_TRANSPOSE = False

        # Level 6
        self.attn1 = AttentionBlock(filters[2], filters[3], filters[3])
        if CONV_TRANSPOSE:
            self.upsample_1 = nn.ConvTranspose2d(filters[3], filters[3], kernel_size=2, stride=2)
        else:
            self.upsample_1 = Upsample_(2)

        # the bias of the up residual out layer is set to true if not followed by a BN layer
        self.up_residual_block1 = ResidualDilatedBlock(filters[3] + filters[2], filters[2],
                                                       stride=1, dilation=1, padding=1,
                                                       bias_out_layer=False)

        # Level 7
        self.attn2 = AttentionBlock(filters[1], filters[2], filters[2])
        if CONV_TRANSPOSE:
            self.upsample_2 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=2, stride=2)
        else:
            self.upsample_2 = Upsample_(2)

        self.up_residual_block2 = ResidualDilatedBlock(filters[2] + filters[1], filters[1],
                                                       stride=1, dilation=1, padding=1,
                                                       bias_out_layer=False)

        # Level 8
        self.attn3 = AttentionBlock(filters[0], filters[1], filters[1])
        if CONV_TRANSPOSE:
            self.upsample_3 = nn.ConvTranspose2d(filters[1], filters[1], kernel_size=2, stride=2)
        else:
            self.upsample_3 = Upsample_(2)

        self.up_residual_block3 = ResidualDilatedBlock(filters[1] + filters[0], filters[0],
                                                       stride=1, dilation=1, padding=1,
                                                       bias_out_layer=False)

        # Choose output kernel size
        if not config.use_submission_masks:
            out_kernel_size = 1
            out_stride = 1
        else:
            out_kernel_size = 16
            out_stride = 16

        # Output
        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], out_channels=1, kernel_size=out_kernel_size, stride=out_stride),
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
        lv5 = self.bridge(lv4)

        # Decoder
        # Level 6
        x6 = self.attn1(lv3, lv5)
        x6 = self.upsample_1(x6)
        x6 = torch.cat([x6, lv3], dim=1)
        x6 = self.up_residual_block1(x6)

        # Level 7
        x7 = self.attn2(lv2, x6)
        x7 = self.upsample_2(x7)
        x7 = torch.cat([x7, lv2], dim=1)
        x7 = self.up_residual_block2(x7)

        # Level 8
        x8 = self.attn3(lv1, x7)
        x8 = self.upsample_3(x8)
        x8 = torch.cat([x8, lv1], dim=1)
        x8 = self.up_residual_block3(x8)

        # Output
        output = self.output_layer(x8)

        return output
