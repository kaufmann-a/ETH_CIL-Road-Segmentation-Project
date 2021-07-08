#!/usr/bin/env python3
# coding: utf8

"""
Handles the folder structure of the different datasets.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"


import os

from source.configuration import Configuration
from source.exceptions.configurationerror import DatasetError
from source.logcreator.logcreator import Logcreator


class DataCollection:
    """
        A data collection folder needs to have the structure:
        +-- data-collection-folder
            +-- original
                +-- images
                +-- masks
            +-- [transformation-folders] e.g. {flip_hor, flip_hor_90, flip_ver, ...}
                +-- images
                +-- masks
    """

    def __init__(self, path, collection_folders):
        """

        :param path: Path to the data collection folders.
        :param collection_folders: Data collection folder list.

        """
        self.path = path
        self.collection_folders = collection_folders

    def get_transformed_images(self):
        """
        Gets images that are transformed from the respective transformation folders.

        :return: images path list, mask path list
        """
        transform_folders = Configuration.get('data_collection.transform_folders')

        transform_folders = [os.path.join(self.path, cur_collection, cur_transformation)
                             for cur_transformation in transform_folders
                             for cur_collection in self.collection_folders
                             if os.path.exists(os.path.join(self.path, cur_collection, cur_transformation))]
        # Read all transformation images
        images_trans, masks_trans = DataCollection.assign_masks_to_images(transform_folders)

        return images_trans, masks_trans

    def get_original_images(self):
        """
        Gets all original images and masks paths that are in the folder "original" in the respective collection folder.

        :return: image path list, mask path list
        """
        collections_folders_orig = [os.path.join(self.path, cur_collection, "original") for cur_collection in
                                    self.collection_folders]
        # Read in original images
        try:
            images_orig, masks_orig = DataCollection.assign_masks_to_images(collections_folders_orig)
        except ValueError:
            raise DatasetError()

        return images_orig, masks_orig

    @staticmethod
    def assign_masks_to_images(imgs_folders):
        """
        Reads the paths of all the images and masks in the provided folder.

        :param imgs_folders: The parent folder containing the "images" and "masks" folders.
        :return: image list, mask list
        """
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


class DataCollectionExperiment:
    """
        The experiment folder needs to have the structure:
        +-- experiment folder
            +-- train
                +-- original
                    +-- images
                    +-- masks
                +-- [transformation-folders] e.g. {flip_hor, flip_hor_90, flip_ver, ...}
                    +-- images
                    +-- masks
            +-- valid
                +-- original
                    +-- images
                    +-- masks
    """
    experiments_dataset_folder = "experiments_dataset"

    def __init__(self, path, collection_folders):
        """

        :param path: Path to the data collection folders.
        :param collection_folders: Data collection folder list.

        """
        self.path = path
        self.collection_folders = collection_folders

    def get_experiment_image_paths(self):
        """
        Gets all images and masks paths of the experiment dataset.

        :return: train image path list, train mask path list, validation image path list, validation mask path list
        """
        collection_folder = os.path.join(self.path, DataCollectionExperiment.experiments_dataset_folder)

        train_folders = os.listdir(os.path.join(collection_folder, "train"))
        train_set_img_folders = [os.path.join(collection_folder, "train", cur_dataset, "images")
                                 for cur_dataset in train_folders]
        train_set_mask_folders = [os.path.join(collection_folder, "train", cur_dataset, "masks")
                                  for cur_dataset in train_folders]

        validation_folders = os.listdir(os.path.join(collection_folder, "valid"))
        val_set_img_folders = [os.path.join(collection_folder, "valid", cur_dataset, "images")
                               for cur_dataset in validation_folders]
        val_set_mask_folders = [os.path.join(collection_folder, "valid", cur_dataset, "masks")
                                for cur_dataset in validation_folders]

        train_set_imgs, train_set_masks = DataCollectionExperiment.generate_exp_set(train_set_img_folders,
                                                                                    train_set_mask_folders)
        val_set_imgs, val_set_masks = DataCollectionExperiment.generate_exp_set(val_set_img_folders,
                                                                                val_set_mask_folders)

        return train_set_imgs, train_set_masks, val_set_imgs, val_set_masks

    @staticmethod
    def generate_exp_set(set_img_folders, set_mask_folders):
        """
        Reads the paths of all the images and masks in the provided folder.

        :param set_img_folders: The folder containing the images.
        :param set_mask_folders: The folder containing the masks.
        :return: image list, mask list
        """
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
