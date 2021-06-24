#!/usr/bin/env python3
# coding: utf8

"""
Main class for inference
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import os
import time
import argparse
import glob
import sys
import torch
from PIL import Image
import numpy as np

from source.configuration import Configuration
from source.engine import Engine
from source.logcreator.logcreator import Logcreator
from source.helpers import filehelper
from source.prediction.prediction import Prediction

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


dir_trainings = "../../trainings"
dirs_prediction = ["exp_lr/20210617-185522-gcdcnn_exp_base_30_50/prediction-20210618-170610",
                   "exp_lr/20210617-204654-unet_exp_base_198_199/prediction-20210618-163841"]
dir_out = "ensemble"

if __name__ == "__main__":

    if False:
        for file in os.listdir(os.path.join(dir_trainings, dirs_prediction[0], "pred-masks-original")):
            if file.endswith(".png"):
                image = np.asarray(Image.open(os.path.join(dir_trainings, dirs_prediction[0], "pred-masks-original", file))).astype(int)
                for i in range(1, len(dirs_prediction)):
                    image2 = np.asarray(Image.open(os.path.join(dir_trainings, dirs_prediction[i], "pred-masks-original", file))).astype(int)
                    image = image + image2
                image = image / len(dirs_prediction)
                if not os.path.exists(os.path.join(dir_trainings, dir_out)):
                    os.makedirs(os.path.join(dir_trainings, dir_out))
                Image.fromarray(image.astype('uint8'), 'RGB').save(os.path.join(dir_trainings, dir_out, file))

    for file in os.listdir(os.path.join(dir_trainings, dir_out)):
        if file.endswith(".png"):
            image = np.asarray(Image.open(os.path.join(dir_trainings, dirs_prediction[0], "pred-masks-original", file))).astype(int)
    mask_probabilistic_list = []


