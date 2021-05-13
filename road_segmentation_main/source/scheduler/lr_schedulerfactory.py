#!/usr/bin/env python3
# coding: utf8

"""
Builds a learning rate scheduler from configuration.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import torch.optim as optim

from source.configuration import Configuration


class LRSchedulerFactory(object):
    optimizer = None

    @staticmethod
    def build(optimizer, last_epoch=-1):
        LRSchedulerFactory.optimizer = optimizer
        scheduler = Configuration.get('training.lr_scheduler.name')
        return getattr(LRSchedulerFactory, scheduler)(LRSchedulerFactory, Configuration.get('training.lr_scheduler'),
                                                      last_epoch)

    def stepLR(self, options, last_epoch=-1):
        return optim.lr_scheduler.StepLR(self.optimizer,
                                         step_size=options.stepLR.step_size,
                                         gamma=options.stepLR.gamma,
                                         last_epoch=last_epoch)

    def multiStepLR(self, options, last_epoch=-1):
        return optim.lr_scheduler.MultiStepLR(self.optimizer,
                                              milestones=options.multiStepLR.milestones,
                                              gamma=options.multiStepLR.gamma,
                                              last_epoch=last_epoch)

    @staticmethod
    def get_members():
        return {
            'stepLR': LRSchedulerFactory.stepLR,
            'multiStepLR': LRSchedulerFactory.multiStepLR
        }
