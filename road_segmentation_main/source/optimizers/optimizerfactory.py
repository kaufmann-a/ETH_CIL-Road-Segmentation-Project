#!/usr/bin/env python3
# coding: utf8

"""
Builds an optimizer from configuration.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import torch.optim as optim
from source.configuration import Configuration


class OptimizerFactory(object):
    model = False

    @staticmethod
    def build(model):
        OptimizerFactory.model = model
        optimizer = Configuration.get('training.optimizer.name')
        return getattr(OptimizerFactory, optimizer)(OptimizerFactory, Configuration.get('training.optimizer'))

    def adam(self, options):
        return optim.Adam(self.model.parameters(), lr=options.lr)


    @staticmethod
    def get_members():
        return {
            'adam': OptimizerFactory.adam
        }
