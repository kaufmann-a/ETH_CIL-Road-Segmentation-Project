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
from source.models.modules import PPM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ResidualDilatedBlock(nn.Module):
    """
    ResidualDilatedBlock - RDB:
    - BatchNorm2d
    - ReLU
    - Conv2d
    - BatchNorm2d
    - ReLU
    - Conv2d
    - BatchNorm2d
    - ReLU
    - Conv2d

    + skip connection:
     - Conv2d: with 1x1 kernel

    Attention: We batch normalize the input but not the output!

    By default the bias=False in the Conv2d layers, since the BN layer after it will "get rid of it":
        https://discuss.pytorch.org/t/any-purpose-to-set-bias-false-in-densenet-torchvision/22067,
        https://arxiv.org/pdf/1502.03167.pdf page 5,
        Why ResNet has bias=False: https://github.com/KaimingHe/deep-residual-networks/issues/10#issuecomment-194037195
    """

    def __init__(self, input_dim, output_dim, stride, padding, dilation=2, bias_out_layer=False,
                 fixed_skip_kernel=False):
        """
        @param bias_out_layer: True = bias for the last Conv2d layer is set to True
        @param fixed_skip_kernel: True = Uses a 1x1 kernel Conv2d layer with fixed weights equal to ones
                                  False = Uses a 1x1 kernel Conv2d layer where the weights are learned
                                  # TODO fixed_skip_kernel does not work! The loss does not change during training!
        """
        super(ResidualDilatedBlock, self).__init__()
        self.fixed_skip_kernel = fixed_skip_kernel
        self.stride = stride

        # dilated convolution
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding, dilation=1, bias=False
            ),

            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=dilation, dilation=dilation, bias=False),

            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=dilation, dilation=dilation, bias=bias_out_layer)
        )

        # identity connection
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(output_dim)  # adding BN here helped against NaN values
        )

    def forward(self, x):
        if self.fixed_skip_kernel:
            identity_kernel = torch.ones(self.conv_skip[0].weight.shape).to(DEVICE)
            identity = F.conv2d(input=x, weight=identity_kernel, stride=self.stride, padding=0, bias=None)
        else:
            identity = self.conv_skip(x)

        out = self.conv_block(x) + identity

        return out


class GlobalContextDilatedCNN(BaseModel):
    """
    Global context dilated convolutional neural network
    """
    name = 'gcdcnn_bn'

    def __init__(self, config):
        super(GlobalContextDilatedCNN, self).__init__()

        in_channel = 3
        filters = config.features  # [64, 128, 256, 512], [8, 16, 32, 64, 128]

        # Encoder
        # level 1
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channel, filters[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channel, filters[0], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(filters[0])
        )

        # RDB blocks
        self.downs = nn.ModuleList()
        for idx in range(0, len(filters) - 1):
            self.downs.append(ResidualDilatedBlock(filters[idx], filters[idx + 1], stride=2, dilation=2, padding=1))

        # Bridge
        BATCH_NORM_INFRONT_PPM = False
        bins = (1, 2, 3, 6)
        ppm = PPM(filters[-1], int(filters[-1] / len(bins)), bins)
        self.bridge = nn.Sequential(
            nn.BatchNorm2d(filters[-1]),
            ppm,
        ) if BATCH_NORM_INFRONT_PPM else ppm

        # Decoder
        self.ups_upsample = nn.ModuleList()
        self.ups_rdb = nn.ModuleList()
        filters[-1] = filters[-1] * 2  # ppm concatenates input with pyramid pooled layers -> doubles channels
        for idx in reversed(range(1, len(filters))):
            self.ups_upsample.append(
                nn.ConvTranspose2d(filters[idx], filters[idx - 1], kernel_size=(2, 2), stride=(2, 2))
            )
            self.ups_rdb.append(
                ResidualDilatedBlock(filters[idx - 1] * 2, filters[idx - 1], stride=1, dilation=1, padding=1,
                                     bias_out_layer=True)
            )

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
        out_down = []

        # Encoder
        x = self.input_layer(x) + self.input_skip(x)
        out_down.append(x)
        for down in self.downs:
            x = down(x)
            out_down.append(x)

        # Bridge
        x = self.bridge(out_down[-1])  # bridge already concatenates last layer with pyramid pooled output!
        out_down[-1] = None

        # Decoder
        for idx in range(0, len(self.ups_upsample)):
            down_idx = len(self.downs) - 1 - idx

            x = self.ups_upsample[idx](x)
            x = torch.cat([x, out_down[down_idx]], dim=1)
            out_down[down_idx] = None
            x = self.ups_rdb[idx](x)

        # Output
        output = self.output_layer(x)

        return output


if __name__ == '__main__':
    """
    Test if the dimensions work out and print model
    """


    class Config:
        use_submission_masks = False
        features = [64, 128, 256, 512]  # [8, 16, 32, 64, 128]


    model = GlobalContextDilatedCNN(config=Config())
    model.to(DEVICE)
    summary(model, input_size=(3, 608, 608), device=DEVICE)
