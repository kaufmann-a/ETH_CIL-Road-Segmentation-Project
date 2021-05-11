#!/usr/bin/env python3
# coding: utf8

"""
Loads and handels training and validation data collections.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import os

import random

from source.configuration import Configuration
from source.data.transformation import get_transformations
from source.logcreator.logcreator import Logcreator
from source.data.dataset import RoadSegmentationDataset


class DataPreparator(object):

    @staticmethod
    def load(path='', main_folder_name='', test_data_count=None):
        if not path:
            path = Configuration.get_path('data_collection.folder', False)
        if not main_folder_name:
            main_folder_name = str(Configuration.get('data_collection.main_folder_name'))

        # split original images into val and train
        val_ratio = Configuration.get('data_collection.validation_ratio', default=0.2)

        originals_list = os.listdir(os.path.join(path, main_folder_name, 'images'))
        originals = [image for image in originals_list if (image.endswith('.png') or image.endswith('.jpg'))]

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
        image_paths_val = []
        mask_paths_val = []

        include_val_transforms = Configuration.get('data_collection.include_val_transforms')
        # include_val_transforms = True

        # paths to images (original and transformed) for training set
        for folder in folders:
            for file in os.listdir(os.path.join(path, folder, "images")):
                filename = os.fsdecode(file)
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    if filename in originals_train:
                        image_paths_train.append(os.path.join(path, folder, "images", filename))
                        mask_paths_train.append(os.path.join(path, folder, "masks", filename))
                    if filename in originals_val:
                        if include_val_transforms:
                            image_paths_val.append(os.path.join(path, folder, "images", filename))
                            mask_paths_val.append(os.path.join(path, folder, "masks", filename))

        if not include_val_transforms:  # only include originals in validation set
            image_paths_val = [os.path.join(path, main_folder_name, 'images', filename) for filename in originals_val]
            mask_paths_val = [os.path.join(path, main_folder_name, 'masks', filename) for filename in originals_val]

        Logcreator.debug("Found %d images for training and %d images for validation."
                         % (len(image_paths_train), len(image_paths_val)))

        if len(image_paths_train) == 0:
            Logcreator.warn("No training files assigned.")
        if len(image_paths_val) == 0:
            Logcreator.warn("No validation files assigned.")

        # test images

        transform = get_transformations()

        foreground_threshold = Configuration.get("training.general.foreground_threshold")
        cropped_image_size = tuple(Configuration.get("training.general.cropped_image_size"))
        use_submission_masks = Configuration.get("training.general.use_submission_masks")

        train_ds = RoadSegmentationDataset(image_paths_train, mask_paths_train, foreground_threshold, transform,
                                           crop_size=cropped_image_size,
                                           use_submission_masks=use_submission_masks)
        val_ds = RoadSegmentationDataset(image_paths_val, mask_paths_val, foreground_threshold, transform,
                                         crop_size=cropped_image_size,
                                         use_submission_masks=use_submission_masks)
        test_ds = RoadSegmentationDataset([], [], foreground_threshold, transform,
                                          crop_size=cropped_image_size,
                                          use_submission_masks=use_submission_masks)

        return train_ds, val_ds, test_ds
