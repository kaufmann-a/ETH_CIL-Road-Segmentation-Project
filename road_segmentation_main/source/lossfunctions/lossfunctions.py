import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from source.configuration import Configuration
from source.logcreator.logcreator import Logcreator

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

class DicePenaltyLoss(DiceLoss):
    def __init__(self):
        super(DiceLoss).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        add_penalty = Configuration.get("training.general.penalty.add_penalty", optional=True, default=False)
        kernel_size = Configuration.get("training.general.penalty.kernel_size", optional=True, default=3)
        lam = Configuration.get("training.general.penalty.lam", optional=True, default=0.5)
        avgPool = torch.nn.AvgPool2d(kernel_size, stride=1, padding=np.floor(kernel_size/2).astype(int))
        mean_predictions = avgPool(inputs)
        mse_loss = torch.nn.MSELoss()
        penalty = mse_loss(inputs, mean_predictions)
        Logcreator.info("Penalty:", penalty)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # TODO put dice coefficent into one static function
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice + lam * penalty


#https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/focal_loss.py
class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(self, alpha=3, gamma=2, ignore_index=None, reduction='mean', **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ['none', 'mean', 'sum']

        # if self.alpha is None:
        #     self.alpha = torch.ones(2)
        # elif isinstance(self.alpha, (list, np.ndarray)):
        #     self.alpha = np.asarray(self.alpha)
        #     self.alpha = np.reshape(self.alpha, (2))
        #     assert self.alpha.shape[0] == 2, \
        #         'the `alpha` shape is not match the number of class'
        # elif isinstance(self.alpha, (float, int)):
        #     self.alpha = np.asarray([self.alpha, 1.0 - self.alpha], dtype=np.float).view(2)

        # else:
        #     raise TypeError('{} not supported'.format(type(self.alpha)))

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()
        if valid_mask is not None:
            pos_mask = pos_mask * valid_mask
            neg_mask = neg_mask * valid_mask

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -torch.sum(pos_weight * torch.log(prob)) / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * torch.sum(neg_weight * F.logsigmoid(-output)) / (torch.sum(neg_weight) + 1e-4)
        loss = pos_loss + neg_loss

        return loss



class FocalTverskyLoss(nn.Module):
    """
    Based on: https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py
    """

    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def tversky_index(self, y_true, y_pred):
        smooth = 1
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.7
        return (true_pos + smooth) / (true_pos + alpha * false_neg + (
                    1 - alpha) * false_pos + smooth)

    def forward(self, inputs, targets, smooth=1):

        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        pt_1 = self.tversky_index(targets, inputs)
        gamma = 0.75
        return torch.pow((1 - pt_1), gamma)


class LogCoshDiceLoss(nn.Module):
    """
    Based on: https://github.com/shruti-jadon/Semantic-Segmentation-Loss-Functions/blob/master/loss_functions.py
    """

    def __init__(self, weight=None, size_average=True):
        super(LogCoshDiceLoss, self).__init__()

    def generalized_dice_coefficient(self, y_true, y_pred):
        smooth = 1 #TODO: Parameterize this
        y_true_f = torch.flatten(y_true)
        y_pred_f = torch.flatten(y_pred)
        intersection = torch.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
                    torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
        return score

    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.generalized_dice_coefficient(y_true, y_pred)
        return loss

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        x = self.dice_loss(targets, inputs)
        return torch.log((torch.exp(x) + torch.exp(-1*x)) / 2.0)


#TODO: Boundary Loss
# https://github.com/LIVIAETS/boundary-loss, http://proceedings.mlr.press/v102/kervadec19a.html
#https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/boundary_loss.py