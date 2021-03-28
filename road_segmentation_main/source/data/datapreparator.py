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

from source.configuration import Configuration
from source.logcreator.logcreator import Logcreator
from source.data.dataset import RoadSegmentationDataset

class DataPreparator(object):

    @staticmethod
    def load(path='', test_data_count=None):
        if not path:
            path = Configuration.get_path('collection.folder', False)
        image_dir = os.path.join(path, 'images')
        groundtruth_dir = os.path.join(path, 'groundtruth')

        images = os.listdir(image_dir)
        groundtruth = os.listdir(groundtruth_dir)
        images = [image for image in images if image.endswith('.png')]
        # groundtruth = [mask for mask in groundtruth if mask.endswith('.png')]

        # TODO: Decide if we should also splitt test data or if we just use test images provided and upload to kaggle always
        test_images = None
        # test_groundtruth = None
        test_frequency = Configuration.get('collection.test_frequency', default=0)
        if not test_data_count:
            test_data_count = Configuration.get('collection.test_data_count', default=0)
        if test_data_count > 0:
            test_data_count = int(test_data_count)
            test_images = images[-test_data_count:]
            images = images[-len(images) - test_data_count:]
            # test_groundtruth = groundtruth[-test_data_count:]
            # groundtruth = groundtruth[-len(groundtruth) - test_data_count:]

        # Split of validation files
        validation_ratio = Configuration.get('collection.validation_ratio', default=0.2)

        validation_images = images[-int(len(images) * validation_ratio):]
        # validation_groundtruth = groundtruth[-int(len(groundtruth) * validation_ratio):]

        training_images = images[:-int(len(images) * validation_ratio)]
        # training_groundtruth = groundtruth[:-int(len(images) * validation_ratio)]

        Logcreator.debug("Found %d images for training and %d images for validation."
                     % (len(training_images), len(validation_images)))
        if test_images:
            Logcreator.debug("Use %d images for tests after every %d epoch."
                         % (len(test_images), test_frequency))

        if len(training_images) == 0:
            Logcreator.warn("No training files assigned.")
        if len(validation_images) == 0:
            Logcreator.warn("No validation files assigned.")

        train_transform = A.Compose(
            [
                A.Resize(height=400, width=400),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )
        val_transforms = A.Compose(
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

        train_ds = RoadSegmentationDataset(image_dir, groundtruth_dir, training_images, train_transform)
        val_ds = RoadSegmentationDataset(image_dir, groundtruth_dir, validation_images, val_transforms)
        test_ds = RoadSegmentationDataset(image_dir,groundtruth_dir, test_images, val_transforms)

        return train_ds, val_ds, test_ds
