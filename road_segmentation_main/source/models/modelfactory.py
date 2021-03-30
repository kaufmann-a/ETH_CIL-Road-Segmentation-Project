#!/usr/bin/env python3
# coding: utf8

"""
Builds a torch model from configuration.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import glob
# Load all models
from os.path import dirname, basename, isfile

from torchsummary import summary

from source.configuration import Configuration
from source.exceptions.configurationerror import ConfigurationError
from source.models.basemodel import BaseModel

modules = glob.glob(dirname(__file__) + "/*.py")
for module in [basename(f)[:-3] for f in modules if
               isfile(f) and not f.endswith('__init__.py') and not f == "modelfactory"]:
    __import__("source.models." + module)


class ModelFactory(object):

    @staticmethod
    def build(print_model=False):
        model_config = Configuration.get('training.model', optional=False)
        all_models = BaseModel.__subclasses__()
        if model_config.name:
            model = [m(model_config) for m in all_models if m.name.lower() == model_config.name.lower()]
            if model and len(model) > 0:
                if print_model:
                    summary(model[0], input_size=(3, 400, 400), device="cpu")
                return model[0]
        raise ConfigurationError('training.model.name')
