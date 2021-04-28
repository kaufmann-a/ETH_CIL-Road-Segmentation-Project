"""
 Provides the image transformations
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transformations():
    transform = A.Compose(
        [
            # A.Resize(height=400, width=400), # commented because we generally do not want to resize
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    return transform
