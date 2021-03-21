import torch
from torch import nn


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

    def forward(self, input, target):
        pred = input.view(-1)
        truth = target.view(-1)

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = self.dice_coefficent(pred, truth)

        return bce_loss + (1 - dice_coef)

    @staticmethod
    def dice_coefficent(pred, truth):
        smooth = 1.0  # 1e-8

        return (2.0 * (pred * truth).double().sum() + 1) / (
                pred.double().sum() + truth.double().sum() + smooth
        )


def get_accuracy(predicted, target):
    num_correct = 0
    num_pixels = 0

    preds = torch.sigmoid(predicted)
    preds = (preds > 0.5).float()

    num_correct += (preds == target).sum()
    num_pixels += torch.numel(preds)

    dice_score = BCEDiceLoss.dice_coefficent(predicted, target)

    accuracy = num_correct / num_pixels

    return accuracy, dice_score
