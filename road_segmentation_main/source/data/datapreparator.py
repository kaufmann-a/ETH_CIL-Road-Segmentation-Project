#!/usr/bin/env python3
# coding: utf8

"""
Loads and handels training and validation data collections.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

from source.configuration import Configuration
from source.logcreator.logcreator import Logcreator
from source.data.dataset import RoadSegmentationDataset


class DataPreparator(object):

    @staticmethod
    def load(path='', test_data_count=None):
        if not path:
            path = Configuration.get_path('data_collection.folder', False)

        # split original images into val and train
        val_ratio = Configuration.get('data_collection.validation_ratio', default=0.2)

        originals_list = os.listdir(os.path.join(path, 'original', 'images'))
        originals = [image for image in originals_list if image.endswith('.png')]

        random.seed(0)
        originals_val = random.sample(originals, int(len(originals) * val_ratio))
        originals_train = [image for image in originals if image not in originals_val]

        # make sure there is no intersection
        if len(list(set(originals_val) & set(originals_train))) > 0:
            Logcreator.warn("Intersection between train and validation set.")

        # get folder names of transformed images
        folders = Configuration.get('data_collection.transform_folders')

        image_paths_train = []
        mask_paths_train = []

        # paths to images (original and transformed) for training set
        for folder in folders:
            for file in os.listdir(os.path.join(path, folder, "images")):
                filename = os.fsdecode(file)
                if filename.endswith(".png"):
                    if filename in originals_train:
                        image_paths_train.append(os.path.join(path, folder, "images", filename))
                        mask_paths_train.append(os.path.join(path, folder, "masks", filename))

        # paths to images (original only) for validation set
        # if we want to use the transformed images for validation too, use the structure from above and include "else"
        image_paths_val = [os.path.join(path, 'original', 'images', filename) for filename in originals_val]
        mask_paths_val = [os.path.join(path, 'original', 'masks', filename) for filename in originals_val]

        Logcreator.debug("Found %d images for training and %d images for validation."
                         % (len(image_paths_train), len(image_paths_val)))

        if len(image_paths_train) == 0:
            Logcreator.warn("No training files assigned.")
        if len(image_paths_val) == 0:
            Logcreator.warn("No validation files assigned.")

        # test images



        transform = A.Compose(
            [
                A.Resize(height=400, width=400),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

        foreground_threshold = Configuration.get("training.general.foreground_threshold")
        train_ds = RoadSegmentationDataset(image_paths_train, mask_paths_train, foreground_threshold, transform)
        val_ds = RoadSegmentationDataset(image_paths_val, mask_paths_val, foreground_threshold, transform)
        test_ds = RoadSegmentationDataset([], [], foreground_threshold, transform)

        return train_ds, val_ds, test_ds
