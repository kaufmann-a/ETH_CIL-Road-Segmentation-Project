#!/usr/bin/env python3
# coding: utf8

"""
Loads and handels training and validation data collections.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import os

import numpy as np

from source.configuration import Configuration
from source.data import transformation
from source.data.datacollection import DataCollection, DataCollectionExperiment
from source.logcreator.logcreator import Logcreator

from source.data.dataset import RoadSegmentationDataset, SimpleToTensorDataset, RoadSegmentationDatasetInference


class DataPreparator(object):

    @staticmethod
    def load_all(engine, path='', is_train=False):
        if not path:
            path = Configuration.get_path('data_collection.folder', False)

        collection_folders = Configuration.get('data_collection.collection_names')

        # detect special experiment dataset
        if DataCollectionExperiment.experiments_dataset_folder in collection_folders:
            data_collection_exp = DataCollectionExperiment(path, collection_folders)

            # get image and respective mask paths
            train_set_imgs, train_set_masks, val_set_imgs, val_set_masks = \
                data_collection_exp.get_experiment_image_paths()

            # combine lists
            images = train_set_imgs + val_set_imgs
            masks = train_set_masks + val_set_masks
        else:
            data_collection = DataCollection(path, collection_folders)

            # get image and respective mask paths
            images_orig, masks_orig = data_collection.get_original_images()
            images_trans, masks_trans = data_collection.get_transformed_images()

            # combine lists
            images = images_orig + images_trans
            masks = masks_orig + masks_trans

        return DataPreparator.get_dataset(engine, images, masks, is_train, name="All")

    @staticmethod
    def load(engine, path=''):
        if not path:
            path = Configuration.get_path('data_collection.folder', False)

        collection_folders = Configuration.get('data_collection.collection_names')

        # detect special experiment dataset
        if DataCollectionExperiment.experiments_dataset_folder in collection_folders:
            train_ds, val_ds = DataPreparator.experiment_run_datasets(engine, path, collection_folders)
            return train_ds, val_ds

        data_collection = DataCollection(path, collection_folders)

        # Get image and respective mask paths
        images_orig, masks_orig = data_collection.get_original_images()
        images_trans, masks_trans = data_collection.get_transformed_images()

        # Get train / validation split
        train_set_images, train_set_masks, val_set_images, val_set_masks = \
            DataPreparator.get_train_validation_split(images_orig, masks_orig, images_trans, masks_trans)

        # Get datasets
        train_ds, val_ds = DataPreparator.get_datasets(engine, train_set_images, train_set_masks,
                                                       val_set_images, val_set_masks)

        return train_ds, val_ds

    @staticmethod
    def load_test(engine, path='', crop_size=(608, 608)):
        if not path:
            path = Configuration.get_path('data_collection.folder', False)

        test_images = [os.path.join(path, filename) for filename in os.listdir(path)]

        transform = transformation.get_transformations(engine=engine, is_train=False)

        return RoadSegmentationDatasetInference(engine, test_images, transform=transform, crop_size=crop_size)

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
    def get_datasets(engine, train_set_images, train_set_masks, val_set_images, val_set_masks):
        train_ds = DataPreparator.get_dataset(engine, train_set_images, train_set_masks,
                                              is_train=True,
                                              name="Training",
                                              compute_stats=True)

        val_ds = DataPreparator.get_dataset(engine, val_set_images, val_set_masks,
                                            is_train=False,
                                            name="Validation")

        return train_ds, val_ds

    @staticmethod
    def get_dataset(engine, images, masks, is_train, name="training", compute_stats=False):
        Logcreator.h1(name, "set contains " + str(len(images)) + " images")
        if len(images) == 0:
            Logcreator.warn("No ", name, "files assigned.")

        # Create dataset
        foreground_threshold = Configuration.get("training.general.foreground_threshold")
        cropped_image_size = tuple(Configuration.get("training.general.cropped_image_size"))
        use_submission_masks = Configuration.get("training.general.use_submission_masks")
        min_road_percentage = Configuration.get("data_collection.min_road_percentage", optional=True, default=0)
        include_overlapping_patches = Configuration.get("data_collection.include_overlapping_patches",
                                                        optional=True, default=True)

        # get image transforms
        image_transforms = transformation.get_transformations(engine=engine, is_train=is_train)

        ds = RoadSegmentationDataset(engine, images, masks, foreground_threshold,
                                     transform=image_transforms,
                                     crop_size=cropped_image_size,
                                     use_submission_masks=use_submission_masks,
                                     min_road_percentage=min_road_percentage,
                                     include_overlapping_patches=include_overlapping_patches)

        if compute_stats:
            mean_after, std_after = transformation.get_mean_std(ds)
            Logcreator.info(f"Mean and std after transformations: mean {mean_after}, std {std_after}")

        return ds

    @staticmethod
    def experiment_run_datasets(engine, path, collection_folders):
        data_collection_exp = DataCollectionExperiment(path, collection_folders)

        train_set_imgs, train_set_masks, val_set_imgs, val_set_masks = \
            data_collection_exp.get_experiment_image_paths()

        train_ds, val_ds = DataPreparator.get_datasets(engine, train_set_imgs, train_set_masks,
                                                       val_set_imgs, val_set_masks)
        return train_ds, val_ds
