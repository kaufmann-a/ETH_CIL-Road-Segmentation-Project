import os

import albumentations as A
import cv2
import numpy as np
from albumentations.augmentations import functional as F


class Rotation(A.DualTransform):
    """
    Rotates the image by a fixed angle of 90deg * factor.
    """

    def __init__(self, factor=1, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.factor = factor

    def apply(self, img, **params):
        return np.ascontiguousarray(np.rot90(img, self.factor))

    def get_params(self):
        return {}

    def apply_to_bbox(self, bbox, **params):
        return F.bbox_rot90(bbox, self.factor, **params)

    def apply_to_keypoint(self, keypoint, **params):
        return F.keypoint_rot90(keypoint, self.factor, **params)

    def get_transform_init_args_names(self):
        return ()


# desired transformations
rotate_90 = A.Compose([Rotation(factor=1, always_apply=True)])
rotate_180 = A.Compose([Rotation(factor=2, always_apply=True)])
rotate_270 = A.Compose([Rotation(factor=3, always_apply=True)])

flip_hor = A.Compose([A.HorizontalFlip(always_apply=True)])
flip_hor_90 = A.Compose([A.HorizontalFlip(always_apply=True),
                         Rotation(factor=1, always_apply=True)])
flip_ver = A.Compose([A.VerticalFlip(always_apply=True)])
flip_ver_90 = A.Compose([A.VerticalFlip(always_apply=True),
                         Rotation(factor=1, always_apply=True)])

crop_random = A.Compose([A.RandomCrop(always_apply=True, width=400, height=400)])
rotate_random = A.Compose([A.Rotate(always_apply=True)])

# TODO is transformations used?
transformations = [flip_hor, flip_ver]

IMAGE_DIR = "../data/training/original/images"
MASK_DIR = "../data/training/original/masks"
DIR = "../data/training"


def transform_and_save(image, mask, name, transform):
    transformed = transform(image=image, mask=mask)
    aug_filename = filename

    aug_image_dir = os.path.join(DIR, name, "images")
    aug_mask_dir = os.path.join(DIR, name, "masks")

    if not os.path.exists(aug_image_dir):
        os.makedirs(os.path.join(DIR, name, "images"))
    if not os.path.exists(aug_mask_dir):
        os.makedirs(os.path.join(DIR, name, "masks"))

    cv2.imwrite(
        filename=os.path.join(aug_image_dir, aug_filename),
        img=transformed["image"],
    )

    cv2.imwrite(
        filename=os.path.join(aug_mask_dir, aug_filename),
        img=transformed["mask"],
    )


if __name__ == '__main__':
    for file in os.listdir(IMAGE_DIR):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            image = cv2.imread(os.path.join(IMAGE_DIR, filename))
            mask = cv2.imread(os.path.join(MASK_DIR, filename))
            print(f"{filename}")

            # apply the transformations
            # rotations
            transform_and_save(image, mask, name="rotate_90", transform=rotate_90)
            transform_and_save(image, mask, name="rotate_180", transform=rotate_180)
            transform_and_save(image, mask, name="rotate_270", transform=rotate_270)

            # flipped image plus rotations
            transform_and_save(image, mask, name="flip_hor", transform=flip_hor)
            transform_and_save(image, mask, name="flip_hor_90", transform=flip_hor_90)
            transform_and_save(image, mask, name="flip_ver", transform=flip_ver)  # flipped + 180
            transform_and_save(image, mask, name="flip_ver_90", transform=flip_ver_90)  # flipped + 270

            # TODO fix random seed
            # random
            transform_and_save(image, mask, name="crop_random", transform=crop_random)
            transform_and_save(image, mask, name="rotate_random", transform=rotate_random)
