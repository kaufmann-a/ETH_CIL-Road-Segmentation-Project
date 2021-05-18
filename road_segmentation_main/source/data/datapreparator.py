#!/usr/bin/env python3
# coding: utf8

"""
Loads and handels training and validation data collections.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import os

import random
import numpy as np

from source.configuration import Configuration
from source.data.transformation import get_transformations
from source.logcreator.logcreator import Logcreator
from source.data.dataset import RoadSegmentationDataset


class DataPreparator(object):

    @staticmethod
    def load(path=''):
        if not path:
            path = Configuration.get_path('data_collection.folder', False)

        collection_folders = Configuration.get('data_collection.collection_names')
        transform_folders = Configuration.get('data_collection.transform_folders')
        val_ratio = Configuration.get('data_collection.validation_ratio', default=0.2)

        collections_folders_orig = [os.path.join(path, cur_collection, "original") for cur_collection in collection_folders]
        transform_folders = [os.path.join(path, cur_collection, cur_transformation)
                             for cur_transformation in transform_folders
                             for cur_collection in collection_folders
                             if os.path.exists(os.path.join(path, cur_collection, cur_transformation))]

        train_set_images_orig = []
        train_set_masks_orig = []

        # Read in original imges
        for orig_folder in collections_folders_orig:
            train_set_images_orig += [os.path.join(orig_folder, "images", img) for img in
                                      os.listdir(os.path.join(orig_folder, "images"))
                                      if img.endswith('.png') or img.endswith('.jpg')]
            train_set_masks_orig += [os.path.join(orig_folder, "masks", img) for img in
                                     os.listdir(os.path.join(orig_folder, "masks"))
                                     if img.endswith('.png') or img.endswith('.jpg')]

        #Create mask for validation set
        train_set_images = []
        train_set_masks = []
        val_set_images = []
        val_set_masks = []
        size_val_set = int(len(train_set_images_orig) * val_ratio)
        mask = np.full(len(train_set_images_orig), False)
        mask[:size_val_set] = True
        np.random.seed(0)
        np.random.shuffle(mask)

        # Define validation set
        for idx, add_to_val_set in enumerate(mask):
            if add_to_val_set:
                val_set_images.append(train_set_images_orig[idx])
                val_set_masks.append(train_set_masks_orig[idx])
            else:
                train_set_images.append(train_set_images_orig[idx])
                train_set_masks.append(train_set_masks_orig[idx])

        train_set_images_trans = []
        train_set_masks_trans = []
        # Read all transformation images
        for transform_folder in transform_folders:
            train_set_images_trans += [os.path.join(transform_folder, "images", img) for img in
                                       os.listdir(os.path.join(transform_folder, "images"))
                                       if img.endswith('.png') or img.endswith('.jpg')]
            train_set_masks_trans += [os.path.join(transform_folder, "masks", img) for img in
                                      os.listdir(os.path.join(transform_folder, "masks"))
                                      if img.endswith('.png') or img.endswith('.jpg')]

        # Add transformations to training set
        for idx, image_path in enumerate(train_set_images_trans):
            split_path = os.path.normpath(image_path).split('\\')
            img_name = split_path[-1]
            collection = split_path[-4]
            red_list = list(filter(lambda j: img_name in j, list(filter(lambda k: collection in k, val_set_images))))
            if len(red_list) == 0:
                train_set_images.append(image_path)
                train_set_masks.append(train_set_masks_trans[idx])

        Logcreator.info("Trainingset contains " + str(len(train_set_images)) + " images")
        Logcreator.info("Validationset constains " + str(len(val_set_images)) + " iamges")

        if len(train_set_images) == 0:
            Logcreator.warn("No training files assigned.")
        if len(val_set_images) == 0:
            Logcreator.warn("No validation files assigned.")

        transform = get_transformations()

        foreground_threshold = Configuration.get("training.general.foreground_threshold")
        cropped_image_size = tuple(Configuration.get("training.general.cropped_image_size"))
        use_submission_masks = Configuration.get("training.general.use_submission_masks")

        train_ds = RoadSegmentationDataset(train_set_images, train_set_masks, foreground_threshold, transform,
                                           crop_size=cropped_image_size,
                                           use_submission_masks=use_submission_masks)
        val_ds = RoadSegmentationDataset(val_set_images, val_set_masks, foreground_threshold, transform,
                                         crop_size=cropped_image_size,
                                         use_submission_masks=use_submission_masks)

        return train_ds, val_ds
