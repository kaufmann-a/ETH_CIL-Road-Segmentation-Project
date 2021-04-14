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
            # TODO is padding = dilation ok?
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=dilation, dilation=dilation, bias=False),

            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            # TODO is padding = dilation ok?
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=dilation, dilation=dilation, bias=bias_out_layer)
        )

        # TODO not clear how they implemented the identity mapping
        # identity connection
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=1, stride=stride, padding=0,
                      bias=bias_out_layer),
            # nn.BatchNorm2d(output_dim) # TODO add BN?
        )

    def forward(self, x):
        if self.fixed_skip_kernel:
            identity_kernel = torch.ones(self.conv_skip[0].weight.shape).to(DEVICE)
            identity = F.conv2d(input=x, weight=identity_kernel, stride=self.stride, padding=0, bias=None)
        else:
            identity = self.conv_skip(x)

        out = self.conv_block(x) + identity

        return out


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
        BATCH_NORM_INFRONT_PPM = False

        # Bridge
        # Level 5
        bins = (1, 2, 3, 6)
        ppm = PPM(filters[3], int(filters[3] / len(bins)), bins)
        self.bridge = nn.Sequential(
            nn.BatchNorm2d(filters[3]),
            ppm,
        ) if BATCH_NORM_INFRONT_PPM else ppm

        # Decoder
        # Accordint to paper:
        #  "Upsampling is done with the transposed convolution of pytorch using kernel size 2 and stride 2"

        # Level 6
        self.upsample_1 = nn.ConvTranspose2d(filters[3] * 2, filters[2], kernel_size=2, stride=2)

        # the bias of the up residual out layer is set to true if not followed by a BN layer
        self.up_residual_block1 = ResidualDilatedBlock(filters[2] * 2, filters[2],
                                                       stride=1, dilation=1, padding=1,
                                                       bias_out_layer=True)

        # Level 7
        self.upsample_2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)

        self.up_residual_block2 = ResidualDilatedBlock(filters[1] * 2, filters[1],
                                                       stride=1, dilation=1, padding=1,
                                                       bias_out_layer=True)

        # Level 8
        self.upsample_3 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)

        self.up_residual_block3 = ResidualDilatedBlock(filters[0] * 2, filters[0],
                                                       stride=1, dilation=1, padding=1,
                                                       bias_out_layer=True)

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
        lv5 = self.bridge(lv4)  # bridge already concatenates level 4 with pyramid pooled output!

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
    model.to(DEVICE)
    summary(model, input_size=(3, 256, 256), device=DEVICE)

    # experimental visualization
    VISUALIZE = False
    if VISUALIZE:
        import hiddenlayer as hl  # need to install IPython

        # Changes so it runs:
        # Line 71 in pytorch_builder.py of the hiddenlayer library to:
        # torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)
        #
        # Upgrade changes so we see the stride and dilation for Conv3x3:
        # Changed line 43/44 in transform.py to:
        # name = self.name or " &gt; ".join([l.title + ("" if not "Conv3x3" == l.title else 's:' + str(l.params['strides']) + 'd:' + str(l.params['dilations'])) for l in matches])
        #                 combo = Node(uid=graph.sequence_id(matches),
        #                              name=name, ...
        x = torch.zeros((2, 3, 256, 256), dtype=torch.float, requires_grad=True).to(DEVICE)

        transforms = [
            hl.transforms.Fold("BatchNorm > Relu > Conv", "BnReluConv"),
            hl.transforms.Fold("Conv > BatchNorm > Relu > Conv", "ConvBnReluConv"),
            hl.transforms.Fold("BnReluConv > BnReluConv > BnReluConv", "RDB"),

            hl.transforms.FoldDuplicates(),
        ]

        graph = hl.build_graph(model, x, transforms=transforms)
        out = graph.build_dot()
        out.render(filename='gc_dcnn', view=True)

        # from torchviz import make_dot
        #
        # x = torch.zeros((2, 3, 256, 256), dtype=torch.float, requires_grad=False)
        # out = model(x)
        # make_dot(out).render("gc_dcnn")
