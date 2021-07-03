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
from source.engine import Engine
from source.logcreator.logcreator import Logcreator
from source.helpers import filehelper
from source.prediction.prediction import Prediction

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def process_args(args):
    if args.run_folder:
        args.workingdir = filehelper.build_abspath(args.run_folder, os.getcwd()) #In case relative path was passed in args

        #Load ConfigFile
        args.configuration = glob.glob(os.path.join(args.workingdir, '*.jsonc')) #Assuming only the config file is of type jsonc
        if len(args.configuration) != 1:
            print("More than one config file found or no config file found at all - abort!")
            sys.exit()
        else:
            args.configuration = args.configuration[0]

        #Get weights
        try:
            args.weights = filehelper.get_latest(os.path.join(args.workingdir, 'weights_checkpoint'), '*best.pth')
        except ValueError:
            print("No 'best' weights-checkpoint fond falling back to normal checkpoint")
            try:
                args.weights = filehelper.get_latest(os.path.join(args.workingdir, 'weights_checkpoint'), '*.pth')
            except ValueError:
                print("No weights found, make sure you use a training with stored weights")
                sys.exit()

        #Init config and logger
        Configuration.initialize(args.configuration, args.workingdir, create_output_train=False, create_output_inf=True)
        Logcreator.initialize(False) #Don't write a logfile
    else:
        print("No folder of a previous training run provided, add one in arguments")


if __name__ == "__main__":
    global config

    parser = argparse.ArgumentParser(
        description="Does prediction on predefined set of images"
    )
    parser.add_argument('--run_folder', default='',
                        type=str, help="Input here the folder path of the training run you want to use for inference")
    parser.add_argument('--predict_on_train', default=False,
                        type=bool, help="If true uses the training data collection to predict.")

    args = parser.parse_args()
    start = time.time()

    process_args(args)

    Logcreator.h1("This is a prediction run with test-images")

    # Init engine
    engine = Engine()
    epoch, train_loss, train_accuracy, val_loss, val_accuracy = engine.load_checkpoints(args.weights)

    # Run predictions

    predictor = Prediction(engine=engine,
                           images=Configuration.get_path('data_collection.test_images_folder'),
                           device=DEVICE,
                           threshold=Configuration.get('inference.general.foreground_threshold'),
                           postprocessing=Configuration.get('inference.postprocessing'),
                           use_original_image_size=Configuration.get('inference.general.use_original_image_size'),
                           enable_postprocessing=Configuration.get('inference.general.enable_postprocessing'),
                           use_submission_masks=Configuration.get("training.general.use_submission_masks"),
                           use_swa_model=Configuration.get("training.general.stochastic_weight_averaging.on")
                           )

    if not args.predict_on_train:
        predictor.predict()
    else:
        predictor.predict_train_images()
