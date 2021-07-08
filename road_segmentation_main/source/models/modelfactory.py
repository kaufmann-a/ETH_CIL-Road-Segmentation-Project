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

from source.configuration import Configuration
from source.exceptions.configurationerror import ConfigurationError
from source.models.basemodel import BaseModel
from source.logcreator.logcreator import Logcreator

modules = glob.glob(dirname(__file__) + "/*.py")
for module in [basename(f)[:-3] for f in modules if
               isfile(f) and not f.endswith('__init__.py') and not f == "modelfactory"]:
    __import__("source.models." + module)


class AttrWrapper(object):
    """
    https://stackoverflow.com/questions/6082625/python-dynamically-add-attributes-to-new-style-class-obj
    """

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def __getattr__(self, n):
        return getattr(self._wrapped, n)


class ModelFactory(object):

    @staticmethod
    def build():
        model_config = Configuration.get('training.model', optional=False)

        # add an attribute to the model configuration
        model_config = AttrWrapper(model_config)
        setattr(model_config, "use_submission_masks", Configuration.get('training.general.use_submission_masks'))

        all_models = BaseModel.__subclasses__()
        if model_config.name:
            model = [m(model_config) for m in all_models if m.name.lower() == model_config.name.lower()]
            if model and len(model) > 0:
                return model[0]
        raise ConfigurationError('training.model.name')
