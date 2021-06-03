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
from source.data import transformation
from source.data.transformation import get_transformations
from source.logcreator.logcreator import Logcreator

from source.data.dataset import RoadSegmentationDataset, SimpleToTensorDataset
from source.exceptions.configurationerror import DatasetError

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

        # Read in original imges
        try:
            train_set_images_orig, train_set_masks_orig = DataPreparator.assign_masks_to_images(collections_folders_orig)
        except ValueError:
            raise DatasetError()

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


        # Read all transformation images
        train_set_images_trans, train_set_masks_trans = DataPreparator.assign_masks_to_images(transform_folders)

        # Add transformations to training set
        for idx, image_path in enumerate(train_set_images_trans):
            split_path = os.path.normpath(image_path).split(os.sep)
            img_name = split_path[-1]
            collection = split_path[-4]
            red_list = list(filter(lambda j: img_name in j, list(filter(lambda k: collection in k, val_set_images))))
            if len(red_list) == 0:
                train_set_images.append(image_path)
                train_set_masks.append(train_set_masks_trans[idx])
            else:
                if Configuration.get('data_collection.include_val_transforms'):
                    val_set_images.append(image_path)
                    val_set_masks.append(train_set_masks_trans[idx])

        Logcreator.info("Trainingset contains " + str(len(train_set_images)) + " images")
        Logcreator.info("Validationset constains " + str(len(val_set_images)) + " iamges")

        if len(train_set_images) == 0:
            Logcreator.warn("No training files assigned.")
        if len(val_set_images) == 0:
            Logcreator.warn("No validation files assigned.")

        # Create datasets
        transform_train = DataPreparator.compute_transformations(train_set_images, is_train=True)
        transform_val = DataPreparator.compute_transformations(train_set_images, is_train=False)

        foreground_threshold = Configuration.get("training.general.foreground_threshold")
        cropped_image_size = tuple(Configuration.get("training.general.cropped_image_size"))
        use_submission_masks = Configuration.get("training.general.use_submission_masks")
        min_road_percentage = Configuration.get("data_collection.min_road_percentage", optional=True, default=0)
        include_overlapping_patches = Configuration.get("data_collection.include_overlapping_patches",
                                                        optional=True, default=True)

        train_ds = RoadSegmentationDataset(train_set_images, train_set_masks, foreground_threshold, transform_train,
                                           crop_size=cropped_image_size,
                                           use_submission_masks=use_submission_masks,
                                           min_road_percentage=min_road_percentage,
                                           include_overlapping_patches=include_overlapping_patches)

        mean_after, std_after = transformation.get_mean_std(train_ds)
        Logcreator.info(f"Mean and std after transformations: mean {mean_after}, std {std_after}")

        val_ds = RoadSegmentationDataset(val_set_images, val_set_masks, foreground_threshold, transform_val,
                                         crop_size=cropped_image_size,
                                         use_submission_masks=use_submission_masks,
                                         min_road_percentage=min_road_percentage,
                                         # TODO should we also remove images in the validation set that do not contain road?
                                         include_overlapping_patches=include_overlapping_patches)

        return train_ds, val_ds

    @staticmethod
    def assign_masks_to_images(imgs_folders):
        imgs = []
        masks = []
        for cur_imgs_folder in imgs_folders:
            imgs_cur_imgs_folder = [os.path.join(cur_imgs_folder, "images", img) for img in
                                              os.listdir(os.path.join(cur_imgs_folder, "images"))
                                              if img.endswith('.png') or img.endswith('.jpg') or img.endswith('.png')]

            found_masks_cur_folder = []
            avaliable_masks_cur_folder = os.listdir(os.path.join(cur_imgs_folder, "masks"))
            imgs_to_delete = []

            for cur_image in imgs_cur_imgs_folder:
                if os.path.basename(cur_image) in avaliable_masks_cur_folder:
                    found_masks_cur_folder.append(
                        os.path.join(cur_imgs_folder, "masks", os.path.basename(cur_image)))
                    imgs_to_delete.append(False)
                else:
                    imgs_to_delete.append(True)
                    Logcreator.warn("Attention, image ", cur_image,
                                    " was found in images but not in masks, it is removed from trainingset")
            imgs_cur_imgs_folder = [img for idx, img in enumerate(imgs_cur_imgs_folder) if
                                              not imgs_to_delete[idx]]

            imgs += imgs_cur_imgs_folder
            masks += found_masks_cur_folder
        return imgs, masks

    @staticmethod
    def compute_transformations(image_paths_train, set_train_norm_statistics=False, is_train=True):
        if set_train_norm_statistics:
            simple_dataset = SimpleToTensorDataset(image_paths_train)

            mean, std = transformation.get_mean_std(simple_dataset)
            Logcreator.info(f"Mean and std on training set: mean {mean}, std {std}")

            transform = transformation.get_transformations(mean=mean, std=std, is_train=is_train)
        else:
            transform = transformation.get_transformations(is_train=is_train)

        return transform
