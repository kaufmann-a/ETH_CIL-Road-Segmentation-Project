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
    def load(engine, path=''):
        if not path:
            path = Configuration.get_path('data_collection.folder', False)

        collection_folders = Configuration.get('data_collection.collection_names')

        if "experiments_dataset" in collection_folders:
            train_ds, val_ds = DataPreparator.experiment_run_datasets(engine, path, collection_folders)
            return train_ds, val_ds

        # Get image and respective mask paths
        images_orig, masks_orig = DataPreparator.get_original_images(collection_folders, path)
        images_trans, masks_trans = DataPreparator.get_transformed_images(collection_folders, path)

        # Get train / validation split
        train_set_images, train_set_masks, val_set_images, val_set_masks = \
            DataPreparator.get_train_validation_split(images_orig, masks_orig, images_trans, masks_trans)

        # Get datasets
        train_ds, val_ds = DataPreparator.get_datasets(engine, train_set_images, train_set_masks,
                                                       val_set_images, val_set_masks)

        return train_ds, val_ds

    @staticmethod
    def get_train_validation_split(images_orig, masks_orig, images_trans, masks_trans):
        val_ratio = Configuration.get('data_collection.validation_ratio', default=0.2)

        # Create mask for validation set
        train_set_images = []
        train_set_masks = []
        val_set_images = []
        val_set_masks = []
        size_val_set = int(len(images_orig) * val_ratio)
        mask = np.full(len(images_orig), False)
        mask[:size_val_set] = True
        np.random.seed(0)
        np.random.shuffle(mask)

        # Define validation set
        for idx, add_to_val_set in enumerate(mask):
            if add_to_val_set:
                val_set_images.append(images_orig[idx])
                val_set_masks.append(masks_orig[idx])
            else:
                train_set_images.append(images_orig[idx])
                train_set_masks.append(masks_orig[idx])

        # Add transformations to training set
        for idx, image_path in enumerate(images_trans):
            split_path = os.path.normpath(image_path).split(os.sep)
            img_name = split_path[-1]
            collection = split_path[-4]
            red_list = list(filter(lambda j: img_name in j, list(filter(lambda k: collection in k, val_set_images))))
            if len(red_list) == 0:
                train_set_images.append(image_path)
                train_set_masks.append(masks_trans[idx])
            else:
                if Configuration.get('data_collection.include_val_transforms'):
                    val_set_images.append(image_path)
                    val_set_masks.append(masks_trans[idx])

        return train_set_images, train_set_masks, val_set_images, val_set_masks

    @staticmethod
    def get_transformed_images(collection_folders, path):
        transform_folders = Configuration.get('data_collection.transform_folders')

        transform_folders = [os.path.join(path, cur_collection, cur_transformation)
                             for cur_transformation in transform_folders
                             for cur_collection in collection_folders
                             if os.path.exists(os.path.join(path, cur_collection, cur_transformation))]
        # Read all transformation images
        train_set_images_trans, train_set_masks_trans = DataPreparator.assign_masks_to_images(transform_folders)

        return train_set_images_trans, train_set_masks_trans

    @staticmethod
    def get_original_images(collection_folders, path):
        """
        Gets all original images and masks paths that are in the folder "original" in the respective collection folder.
        A data collection folder needs to have the structure:
        +-- data-collection-folder
            +-- original
                +-- images
                +-- masks

        :param collection_folders: Data collection folder list.
        :param path: Path to the data collection folders.

        :return: image path list, mask path list
        """
        collections_folders_orig = [os.path.join(path, cur_collection, "original") for cur_collection in
                                    collection_folders]
        # Read in original images
        try:
            images_orig, masks_orig = DataPreparator.assign_masks_to_images(collections_folders_orig)
        except ValueError:
            raise DatasetError()

        return images_orig, masks_orig

    @staticmethod
    def get_datasets(engine, train_set_images, train_set_masks, val_set_images, val_set_masks):
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
        train_ds = RoadSegmentationDataset(engine, train_set_images, train_set_masks, foreground_threshold, transform_train,
                                           crop_size=cropped_image_size,
                                           use_submission_masks=use_submission_masks,
                                           min_road_percentage=min_road_percentage,
                                           include_overlapping_patches=include_overlapping_patches)
        mean_after, std_after = transformation.get_mean_std(train_ds)
        Logcreator.info(f"Mean and std after transformations: mean {mean_after}, std {std_after}")
        val_ds = RoadSegmentationDataset(engine, val_set_images, val_set_masks, foreground_threshold, transform_val,
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
    def experiment_run_datasets(engine, path, collection_folders):
        collection_folder = os.path.join(path, collection_folders[0])

        train_folders = os.listdir(os.path.join(collection_folder, "train"))

        train_set_img_folders = [os.path.join(collection_folder, "train", cur_dataset, "images") for cur_dataset in
                                 train_folders]
        train_set_mask_folders = [os.path.join(collection_folder, "train", cur_dataset, "masks") for cur_dataset in
                                  train_folders]

        validation_folders = os.listdir(os.path.join(collection_folder, "valid"))

        val_set_img_folders = [os.path.join(collection_folder, "valid", cur_dataset, "images") for cur_dataset in
                               validation_folders]
        val_set_mask_folders = [os.path.join(collection_folder, "valid", cur_dataset, "masks") for cur_dataset in
                                validation_folders]

        train_set_imgs, train_set_masks = DataPreparator.generate_exp_set(train_set_img_folders, train_set_mask_folders)
        val_set_imgs, val_set_masks = DataPreparator.generate_exp_set(val_set_img_folders, val_set_mask_folders)

        train_ds, val_ds = DataPreparator.get_datasets(engine, train_set_imgs, train_set_masks, val_set_imgs, val_set_masks)
        return train_ds, val_ds

    @staticmethod
    def generate_exp_set(set_img_folders, set_mask_folders):
        set_imgs = []
        set_masks = []
        for idx, folder in enumerate(set_img_folders):
            cur_img_paths = []
            cur_mask_paths = []
            image_names_cur_folder = os.listdir(folder)
            available_masks_cur_folder = os.listdir(set_mask_folders[idx])
            for img in image_names_cur_folder:
                if img in available_masks_cur_folder:
                    cur_img_paths.append(os.path.join(folder, img))
                    cur_mask_paths.append(os.path.join(set_mask_folders[idx], img))
                else:
                    Logcreator.warn("Attention, image ", img,
                                    " was found in images but not in masks, it is removed from trainingset")
            set_imgs += cur_img_paths
            set_masks += cur_mask_paths
        return set_imgs, set_masks

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
