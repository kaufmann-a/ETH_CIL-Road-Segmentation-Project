import albumentations as A
import numpy as np
import cv2
import os

# TODO add desired transformations here
transform = A.Compose([A.HorizontalFlip()])

IMAGE_DIR = "../../data/training/images"
MASK_DIR = "../../data/training/groundtruth"
AUG_IMAGE_DIR = "../../data/training/augmentations/images"
AUG_MASK_DIR = "../../data/training/augmentations/masks"

for file in os.listdir(IMAGE_DIR):
    filename = os.fsdecode(file)
    if filename.endswith(".png"):
        image = cv2.imread(os.path.join(IMAGE_DIR, filename))
        mask = cv2.imread(os.path.join(MASK_DIR, filename))

        transformed = transform(image=image, mask=mask)
        aug_filename = "aug_" + filename

        cv2.imwrite(
            filename=os.path.join(AUG_IMAGE_DIR, aug_filename),
            img=transformed["image"],
        )

        cv2.imwrite(
            filename=os.path.join(AUG_MASK_DIR, aug_filename),
            img=transformed["mask"],
        )