import os

import albumentations as A
import cv2
import numpy as np
from albumentations.augmentations import functional as F



crop = A.Compose([A.CenterCrop(always_apply=True, width=400, height=400)])

IMAGE_DIR = "../data/training/github_jkf/images"
MASK_DIR = "../data/training/github_jkf/masks"
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
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image = cv2.imread(os.path.join(IMAGE_DIR, filename))
            mask = cv2.imread(os.path.join(MASK_DIR, filename))

            transform_and_save(image, mask, name="github_jkf_cropped_400", transform=crop)
