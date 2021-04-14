import torch
from torch import nn


class DiceBCELoss(nn.Module):
    """
    Expects inputs to be non probabilistic.
    """

    def __init__(self, weight=None, size_average=None):
        super().__init__()
        self.weight = weight
        self.size_average = size_average

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCEWithLogitsLoss(self.weight, self.size_average)(pred, truth).double()

        # Dice Loss
        dice_coef = self.dice_coefficent(pred, truth)

        return bce_loss + (1 - dice_coef)

    @staticmethod
    def dice_coefficent(pred, truth):
        pred = torch.sigmoid(pred)
        smooth = 1.0  # 1e-8

        return (2.0 * (pred * truth).double().sum() + smooth) / (pred.double().sum() + truth.double().sum() + smooth)


class DiceLoss(nn.Module):
    """
    Expects inputs to be non probabilistic.

    Based on: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch
    """

    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # TODO put dice coefficent into one static function
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice
