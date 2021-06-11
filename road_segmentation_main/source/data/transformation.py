"""
 Provides the image transformations
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import os
import sys

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

from source.configuration import Configuration
from source.logcreator.logcreator import Logcreator

MEAN_TRAIN_IMAGES = [0.3151, 0.3108, 0.2732]
STD_TRAIN_IMAGES = [0.1721, 0.1637, 0.1632]


def get_mean_std(dataset, debug=False):
    """
    Compute the mean an std over a dataset of images.

    source: https://stackoverflow.com/questions/60101240/finding-mean-and-standard-deviation-across-image-channels-pytorch

    :param dataset:
    :return: mean, std
    """

    loader = DataLoader(dataset, batch_size=8, num_workers=8)
    loop = tqdm(loader, file=sys.stdout, desc="Compute mean and std")

    nimages = 0
    mean = 0.0
    var = 0.0

    for images in loop:
        if type(images) is list and len(images) == 2:  # case when dataset returns image and mask
            # extract image
            images = images[0]

        # Rearrange batch to be the shape of [B, C, W * H]
        images = images.view(images.size(0), images.size(1), -1)
        # Update total number of images
        nimages += images.size(0)
        # Compute mean and std here
        mean += images.mean(2).sum(0)
        var += images.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    if debug:
        print("mean", mean)
        print("std", std)

    return mean, std


def get_default_config():
    try:
        cfg = Configuration.get("data_collection.augmentations")
    except EnvironmentError as exception:
        # assert that only this error is ignored
        assert (exception.args[0] == "Invalid key or wrong key specification")

        class AugmentationConfig(object):
            list = list()

        cfg = AugmentationConfig()  # return empty list if no augmentations specified

    return cfg


def get_transformations(mean=None, std=None, use_train_statistics=False, is_train=False):
    if mean is None:
        mean = [0.0, 0.0, 0.0]
    if std is None:
        std = [1.0, 1.0, 1.0]

    cfg = get_default_config()

    if use_train_statistics or (hasattr(cfg, "use_train_statistics") and cfg.use_train_statistics):
        mean = MEAN_TRAIN_IMAGES
        std = STD_TRAIN_IMAGES

    transform_list = list()

    if is_train:
        # dynamically load augmentations according to config
        for method in cfg.list:
            aug_class = eval("A." + method["name"])
            if "params" in method:
                aug_instance = aug_class(**method["params"])
            else:
                aug_instance = aug_class()

            transform_list.append(aug_instance)

    if hasattr(cfg, "use_histogram_matching") and cfg.use_histogram_matching:
        test_img_dir = os.path.join(os.getenv("DATA_COLLECTION_DIR"), "../test_images")
        test_imgs = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir)]
        transform_list += [
            A.HistogramMatching(
                blend_ratio=(0.8, 1.0),
                reference_images=test_imgs,
                always_apply=True)
        ]

    transform_list += [
        # A.Resize(height=400, width=400), # commented because we generally do not want to resize
        A.Normalize(
            mean=mean,
            std=std,
            max_pixel_value=255.0,
        ),
        ToTensorV2()
    ]

    # build string for logging
    transformations_str = "\n"
    for t in transform_list:
        transformations_str += str(t) + "\n"

    Logcreator.info(f"Applied transformations (is_train={is_train}):", transformations_str)

    transforms = A.Compose(transforms=transform_list)

    return transforms


if __name__ == '__main__':
    print("augmentation test run")
    Configuration.initialize("../../configurations/default.jsonc", os.getcwd(),
                             create_output_train=False,
                             create_output_inf=False)

    from PIL import Image
    import numpy
    import matplotlib.pyplot as plt
    import cv2

    image = Image.open("../../../data/training/eth_dataset/original/images/satImage_001.png").convert("RGB")
    mask = Image.open("../../../data/training/eth_dataset/original/masks/satImage_001.png").convert("RGB")

    LOAD_FROM_CONFIG = False
    if LOAD_FROM_CONFIG:
        transform = get_transformations(is_train=True)
    else:
        transform_list = []
        transform_list += [
            A.HistogramMatching(
                reference_images=["../../../data/test_images/test_7.png"],
                always_apply=True)
        ]

        transform_list += [
            # A.CLAHE(p=1.0),
            # A.HueSaturationValue(p=1.0),
            # A.ColorJitter(p=1.0),
            # A.ChannelShuffle(p=1.0),
            # A.ChannelDropout(p=1.0),
            A.ShiftScaleRotate(p=1.0,
                               shift_limit=0.1,
                               scale_limit=0.625,
                               rotate_limit=45,
                               interpolation=cv2.INTER_LINEAR),  # cv2.INTER_CUBIC
            # A.GaussianBlur(p=1.0),
            # A.RandomFog(fog_coef_upper=0.5, p=1.0),
            # A.RandomContrast(p=1.0),
            # A.MedianBlur(),
            # A.GaussNoise(p=1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2()
        ]

        transform = A.Compose(transforms=transform_list)

    for i in range(0, 10):
        augmentations = transform(image=numpy.array(image), mask=numpy.array(mask))
        aug_image = augmentations["image"]
        aug_mask = augmentations["mask"]

        fig, ax = plt.subplots(nrows=1, ncols=2)

        ax[0].imshow(aug_image.permute(1, 2, 0))
        ax[1].imshow(aug_mask)

        plt.show()
