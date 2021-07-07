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

from source.configuration import Configuration
from source.logcreator.logcreator import Logcreator
from source.hough_transforms.hough_transforms import HoughTransforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Creates Hough Transforms on Predictions"
    )
    parser.add_argument('--prediction_folder', default='',
                        type=str, help="Input here the folder path of the predictions you want to create hough transforms of")

    args = parser.parse_args()
    start = time.time()

    Logcreator.h1("This creates hough transform lines")

    ht = HoughTransforms()
    ht.create_line_images(directory = args.prediction_folder)
