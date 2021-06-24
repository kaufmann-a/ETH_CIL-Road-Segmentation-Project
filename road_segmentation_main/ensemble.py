#!/usr/bin/env python3
# coding: utf8

"""
Class for prediction of a set of images
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import os

import argparse
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from source.configuration import Configuration
from source.data.dataset import EnsembleDataset
from source.helpers import predictionhelper, imagesavehelper, argumenthelper
from source.logcreator.logcreator import Logcreator

PATCH_SIZE = 16

class Ensemble():
    def __init__(self, prediction_dirs):
        self.prediction_dirs = prediction_dirs

    def load_prediction_masks(self, prediction_dirs):
        """
        prediction_dir: list of directories of predicted masks

        return: list of lists of images, image numbers
        """
        Logcreator.info("Loading Images")
        Logcreator.info("Found %d prediction datasets" % len(prediction_dirs))
        out_image_lists = [[] for i in range(len(prediction_dirs))]
        image_number_list = []

        Logcreator.info("Found %d images" % len(os.listdir(os.path.join(dir_trainings, prediction_dirs[0], 'pred-masks-original'))))
        for file in os.listdir(os.path.join(dir_trainings, prediction_dirs[0], 'pred-masks-original')):
            filename = os.fsdecode(file)
            if filename.endswith(".png"):
                # get image number
                image_number_list.append([int(s) for s in filename[:-4].split("_") if s.isdigit()][0])
                # get images
                for i in range(0, len(prediction_dirs)):
                    image = Image.open(os.path.join(dir_trainings, prediction_dirs[i], 'pred-masks-original', filename)).getchannel(0)
                    out_image_lists[i].append(image)

        return out_image_lists, image_number_list

    def ensemble(self, mode="binary"):
        # load the images
        image_lists, image_number_list = self.load_prediction_masks(prediction_dirs=self.prediction_dirs)
        # create the dataset and dataloader (returns average masks)
        dataset = EnsembleDataset(image_lists=image_lists)
        loader = DataLoader(dataset, batch_size=8, num_workers=2, shuffle=False)

        mask_list = []
        Logcreator.info("Iterating over Dataset")
        loop = tqdm(loader)
        for idx, x in enumerate(loop):
            for i in range(x.shape[0]):
                if mode == "binary":
                    img = (x[i] >= 0.5).float()
                if mode == "probabilistic":
                    img = x[i]
                else:
                    img = x[i]
                mask_list.append(img[0])

        Logcreator.info("Write predictions to submission file")
        predictionhelper.images_to_submission_file(mask_list, image_number_list,
                                                   foreground_threshold=0.25,
                                                   folder=Configuration.output_directory,
                                                   file_prefix="",
                                                   patch_size=PATCH_SIZE)
        Logcreator.info("Save masks as images")
        mask_binary_list = imagesavehelper.save_masks_as_images(mask_list, image_number_list,
                                                                folder=Configuration.output_directory,
                                                                is_prob=True,
                                                                pixel_threshold=0.25,
                                                                save_submission_img=True,
                                                                folder_prefix="")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Averages multiple predictions")
    parser.add_argument('--configuration', default='./configurations/ensemble-default.jsonc', type=str, help="Training configuration.")
    parser.add_argument('--workingdir', default=os.getcwd(), type=str, help="Working directory (default: current directory).")

    args = argumenthelper.parse_args(parser)

    Configuration.initialize(args.configuration, args.workingdir, create_output_train=True, create_output_inf=False)
    Logcreator.initialize()

    Logcreator.h1("Ensemble")
    Logcreator.info("Environment: %s" % Configuration.get('environment.name'))

    # directories of predictions used for averaging
    dir_trainings = Configuration.get('environment.output_path')

    # these directories have to be in "dir_trainings"
    dirs_prediction = Configuration.get('dirs_prediction')
    mode = Configuration.get("mode")
    voting_threshold = Configuration.get("voting_threshold")

    ens = Ensemble(prediction_dirs=dirs_prediction)
    ens.ensemble(mode=mode)

