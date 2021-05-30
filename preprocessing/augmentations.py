import os
import random
import sys

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


def transform_and_save(image, mask, name, transform, directory, filename):
    transformed = transform(image=image, mask=mask)

    aug_image_dir = os.path.join(directory, name, "images")
    aug_mask_dir = os.path.join(directory, name, "masks")

    if not os.path.exists(aug_image_dir):
        os.makedirs(aug_image_dir)
    if not os.path.exists(aug_mask_dir):
        os.makedirs(aug_mask_dir)

    cv2.imwrite(
        filename=os.path.join(aug_image_dir, filename),
        img=transformed["image"],
    )

    cv2.imwrite(
        filename=os.path.join(aug_mask_dir, filename),
        img=transformed["mask"],
    )


def apply_all_transformations(directory):
    image_dir = os.path.join(directory, "original", "images")
    mask_dir = os.path.join(directory, "original", "masks")

    for file in os.listdir(image_dir):
        filename = os.fsdecode(file)
        if filename.endswith(".png") or filename.endswith(".jpg"):

            try:
                image = cv2.imread(os.path.join(image_dir, filename))
                mask = cv2.imread(os.path.join(mask_dir, filename))
                print(f"{filename}")

                # apply the transformations
                # rotations
                transform_and_save(image, mask, name="rotate_90", transform=rotate_90, directory=directory, filename=filename)
                transform_and_save(image, mask, name="rotate_180", transform=rotate_180, directory=directory, filename=filename)
                transform_and_save(image, mask, name="rotate_270", transform=rotate_270, directory=directory, filename=filename)

                # flipped image plus rotations
                transform_and_save(image, mask, name="flip_hor", transform=flip_hor, directory=directory, filename=filename)
                transform_and_save(image, mask, name="flip_hor_90", transform=flip_hor_90, directory=directory, filename=filename)
                transform_and_save(image, mask, name="flip_ver", transform=flip_ver, directory=directory, filename=filename)  # flipped + 180
                transform_and_save(image, mask, name="flip_ver_90", transform=flip_ver_90, directory=directory, filename=filename)  # flipped + 270

                # fix random seed
                random.seed(17)

                # random
                transform_and_save(image, mask, name="crop_random", transform=crop_random, directory=directory, filename=filename)
                transform_and_save(image, mask, name="rotate_random", transform=rotate_random, directory=directory, filename=filename)
            except ValueError as verr:
                print("Fault message: ", verr.args)
                print('Image ', filename, " could not be transformed")
            except TypeError as tperr:
                print("Fault message: ", tperr.args)
                print('Image ', filename, " could not be transformed")
    print("Finished dataset")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        DIR = sys.argv[1]
    else:
        DIR = "../data/training"
    print("Now crawling thorugh: ", DIR)
    datasets_dir = ["eth_dataset", "jkfrie", "matejsladek", "ottawa", "alessiapacca", "osm_roadtracer"]

    for dataset in datasets_dir:
        print("===============")
        print(dataset)
        print("---------------")
        directory = os.path.join(DIR, dataset)
        apply_all_transformations(directory)


