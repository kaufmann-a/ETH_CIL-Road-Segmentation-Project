"""
 Provides the image transformations
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import albumentations as A
from albumentations.pytorch import ToTensorV2

MEAN_TRAIN_IMAGES = [0.3151, 0.3108, 0.2732]
STD_TRAIN_IMAGES = [0.1721, 0.1637, 0.1632]


def get_transformations(use_train_statistics=False):
    if use_train_statistics:
        mean = MEAN_TRAIN_IMAGES
        std = STD_TRAIN_IMAGES
    else:
        mean = [0.0, 0.0, 0.0],
        std = [1.0, 1.0, 1.0],

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
