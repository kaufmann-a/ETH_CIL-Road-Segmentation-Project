"""
 Based on: https://github.com/rishikksh20/ResUnet
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ASPP(nn.Module):
    """
    Based on: https://github.com/rishikksh20/ResUnet/blob/master/core/modules.py
    """

    def __init__(self, in_dims, out_dims, rate=[6, 12, 18]):
        super(ASPP, self).__init__()

        self.aspp_block1 = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, 3, stride=1, padding=rate[0], dilation=rate[0]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block2 = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, 3, stride=1, padding=rate[1], dilation=rate[1]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )
        self.aspp_block3 = nn.Sequential(
            nn.Conv2d(in_dims, out_dims, 3, stride=1, padding=rate[2], dilation=rate[2]),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_dims),
        )

        self.output = nn.Conv2d(len(rate) * out_dims, out_dims, 1)
        self._init_weights()

    def forward(self, x):
        x1 = self.aspp_block1(x)
        x2 = self.aspp_block2(x)
        x3 = self.aspp_block3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.output(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Upsample_(nn.Module):
    def __init__(self, scale=2):
        super(Upsample_, self).__init__()

        self.upsample = nn.Upsample(mode="bilinear", scale_factor=scale)

    def forward(self, x):
        return self.upsample(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_encoder, input_decoder, output_dim):
        super(AttentionBlock, self).__init__()

        self.conv_encoder = nn.Sequential(
            nn.BatchNorm2d(input_encoder),
            nn.ReLU(),
            nn.Conv2d(input_encoder, output_dim, 3, padding=1),
            nn.MaxPool2d(2, 2),
        )

        self.conv_decoder = nn.Sequential(
            nn.BatchNorm2d(input_decoder),
            nn.ReLU(),
            nn.Conv2d(input_decoder, output_dim, 3, padding=1),
        )

        self.conv_attn = nn.Sequential(
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, 1, 1),
        )

    def forward(self, x1, x2):
        out = self.conv_encoder(x1) + self.conv_decoder(x2)
        out = self.conv_attn(out)
        return out * x2


class PPM(nn.Module):
    """
    Pyramid Pooling Module - PPM
    Based on: https://github.com/hszhao/semseg/blob/master/model/pspnet.py
    """

    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class ASPP_new(nn.Module):
    """
    Atrous Spatial Pyramid Pooling - APP

    Based on https://arxiv.org/pdf/1802.02611v3.pdf,
             https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/aspp.py,
    """

    def __init__(self, in_dims, out_dims, use_bn_relu_out=False, use_global_avg_pooling=True):
        super(ASPP_new, self).__init__()

        self.out_dims = out_dims
        self.use_global_avg_pooling = use_global_avg_pooling

        kernels = [1, 3, 3, 3]
        dilation = [1, 6, 12, 18]

        # hardcoded blocks so that _init_weights() works (probably there is a better method)
        self.aspp_block_1 = self.get_aspp_block(in_dims, out_dims, kernels[0], 0, dilation[0])
        self.aspp_block_2 = self.get_aspp_block(in_dims, out_dims, kernels[1], dilation[1], dilation[1])
        self.aspp_block_3 = self.get_aspp_block(in_dims, out_dims, kernels[2], dilation[2], dilation[2])
        self.aspp_block_4 = self.get_aspp_block(in_dims, out_dims, kernels[3], dilation[3], dilation[3])

        nr_blocks = len(dilation)

        if self.use_global_avg_pooling:
            self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                                 nn.Conv2d(in_dims, out_dims,
                                                           kernel_size=(1, 1), stride=(1, 1), bias=False),
                                                 nn.BatchNorm2d(out_dims),
                                                 nn.ReLU())
            nr_blocks += 1

        if use_bn_relu_out:
            self.output_block = nn.Sequential(
                nn.Conv2d(nr_blocks * out_dims, out_dims, kernel_size=(1, 1), bias=False),
                nn.BatchNorm2d(out_dims),
                nn.ReLU(),
                nn.Dropout(0.5)
            )
        else:
            self.output_block = nn.Sequential(
                nn.Conv2d(nr_blocks * out_dims, out_dims, kernel_size=(1, 1), bias=False),
                nn.Dropout(0.5)
            )

        self._init_weights()

    @staticmethod
    def get_aspp_block(in_dims, out_dims, kernel_size, padding, dilation):
        return nn.Sequential(
            nn.Conv2d(in_dims, out_dims,
                      kernel_size=(kernel_size, kernel_size),
                      stride=(1, 1),
                      padding=(padding, padding),
                      dilation=(dilation, dilation),
                      bias=False),
            nn.BatchNorm2d(out_dims),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.aspp_block_1(x)
        x2 = self.aspp_block_2(x)
        x3 = self.aspp_block_3(x)
        x4 = self.aspp_block_4(x)

        if self.use_global_avg_pooling:
            x_gap = self.global_avg_pool(x)
            x_gap = F.interpolate(x_gap, size=x1.size()[2:], mode='bilinear', align_corners=True)

            out = torch.cat([x1, x2, x3, x4, x_gap], dim=1)
        else:
            out = torch.cat([x1, x2, x3, x4], dim=1)

        return self.output_block(out)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
