#!/usr/bin/env python3
# coding: utf8

"""
Builds a torch loss function from configuration.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import torch.nn as nn

from source.configuration import Configuration
from source.lossfunctions.lossfunctions import DiceLoss, DiceBCELoss, BinaryFocalLoss

class LossFunctionFactory(object):
    model = False

    @staticmethod
    def build(model):
        LossFunctionFactory.model = model
        loss_function = Configuration.get(
            'training.loss_function', optional=False)
        return getattr(LossFunctionFactory, loss_function)()

    @staticmethod
    def bce_with_logits_loss():
        """
        Uses intern a sigmoid to get probabilities.
        """
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def dice_bce_loss():
        return DiceBCELoss()

    @staticmethod
    def dice_loss():
        return DiceLoss()

    @staticmethod
    def binary_focal_loss():
        return BinaryFocalLoss()

    @staticmethod
    def get_members():
        return {
            'bce_with_logits_loss': LossFunctionFactory.bce_with_logits_loss,
            'dice_bce_loss': LossFunctionFactory.dice_bce_loss,
            'dice_loss': LossFunctionFactory.dice_loss,
            'binary_focal_loss': LossFunctionFactory.binary_focal_loss,
        }