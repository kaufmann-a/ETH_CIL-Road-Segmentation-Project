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
    def load(path='', main_folder_name='', test_data_count=None):
        if not path:
            path = Configuration.get_path('data_collection.folder', False)
        # if not main_folder_name:
        #     main_folder_name = str(Configuration.get('data_collection.main_folder_name'))
        transform_folders = Configuration.get('data_collection.transform_folders')
        val_ratio = Configuration.get('data_collection.validation_ratio', default=0.2)
        include_val_transforms = Configuration.get('data_collection.include_val_transforms')

        collections_folders_orig = [os.path.join(path, cur_collection, "original") for cur_collection in os.listdir(path)]
        transform_folders = [os.path.join(path, cur_collection, cur_transformation)
                             for cur_transformation in transform_folders
                             for cur_collection in os.listdir(path)
                             if os.path.exists(os.path.join(path, cur_collection, cur_transformation))]

        # originals_list = os.listdir(os.path.join(path, main_folder_name, 'images'))
        # originals = [image for image in originals_list if (image.endswith('.png') or image.endswith('.jpg'))]

        train_set_images_orig = []
        train_set_masks_orig = []
        train_set_images_trans = []
        train_set_masks_trans = []
        for orig_folder in collections_folders_orig:
            train_set_images_orig += [os.path.join(orig_folder, "images", img) for img in
                                      os.listdir(os.path.join(orig_folder, "images"))
                                      if img.endswith('.png') or img.endswith('.jpg')]
            train_set_masks_orig += [os.path.join(orig_folder, "masks", img) for img in
                                     os.listdir(os.path.join(orig_folder, "masks"))
                                     if img.endswith('.png') or img.endswith('.jpg')]
        for transform_folder in transform_folders:
            train_set_images_trans += [os.path.join(transform_folder, "images", img) for img in
                                       os.listdir(os.path.join(transform_folder, "images"))
                                       if img.endswith('.png') or img.endswith('.jpg')]
            train_set_masks_trans += [os.path.join(transform_folder, "masks", img) for img in
                                      os.listdir(os.path.join(transform_folder, "masks"))
                                      if img.endswith('.png') or img.endswith('.jpg')]

        train_set_images = []
        train_set_masks = []
        val_set_images = []
        val_set_masks = []
        if include_val_transforms:
            train_set_images_orig += train_set_images_trans
            train_set_masks_orig += train_set_masks_trans

        size_val_set = int(len(train_set_images_orig) * val_ratio)
        mask = np.full(len(train_set_images_orig), False)
        mask[:size_val_set] = True
        np.random.seed(0)
        np.random.shuffle(mask)

        for idx, add_to_val_set in enumerate(mask):
            if add_to_val_set:
                val_set_images.append(train_set_images_orig[idx])
                val_set_masks.append(train_set_masks_orig[idx])
            else:
                train_set_images.append(train_set_images_orig[idx])
                train_set_masks.append(train_set_masks_orig[idx])

        if not include_val_transforms: #wrong, check val set orig if img contained there if not add to set
            train_set_images += train_set_images_trans
            train_set_masks += train_set_masks_trans


        if len(train_set_images) == 0:
            Logcreator.warn("No training files assigned.")
        if len(val_set_images) == 0:
            Logcreator.warn("No validation files assigned.")

        # test images

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
