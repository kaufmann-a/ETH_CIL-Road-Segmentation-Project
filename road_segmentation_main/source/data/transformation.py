"""
 Provides the image transformations
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import sys

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def get_transformations(mean=None, std=None, use_train_statistics=False):
    if mean is None:
        mean = [0.0, 0.0, 0.0]
    if std is None:
        std = [1.0, 1.0, 1.0]

    if use_train_statistics:
        mean = MEAN_TRAIN_IMAGES
        std = STD_TRAIN_IMAGES

    transform = A.Compose(
        [
            # A.Resize(height=400, width=400), # commented because we generally do not want to resize
            A.Normalize(
                mean=mean,
                std=std,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    return transform
