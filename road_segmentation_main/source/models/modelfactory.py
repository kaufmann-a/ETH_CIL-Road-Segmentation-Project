#!/usr/bin/env python3
# coding: utf8

"""
Builds a torch model from configuration.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import importlib

from source.configuration import Configuration
from source.exceptions.configurationerror import ConfigurationError
from source.models.basemodel import BaseModel

# Load all models
from os.path import dirname, basename, isfile
import glob
modules = glob.glob(dirname(__file__)+"/*.py")
for module in [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py') and not f == "modelfactory"]:
    __import__("source.models." + module)

class ModelFactory(object):

    @staticmethod
    def build():
        model_config = Configuration.get('training.model', optional=False)
        all_models = BaseModel.__subclasses__()
        if model_config.name:
            model = [m(model_config) for m in all_models if m.name.lower() == model_config.name.lower()]
            if model and len(model) > 0:
                return model[0]
        raise ConfigurationError('training.model.name')
