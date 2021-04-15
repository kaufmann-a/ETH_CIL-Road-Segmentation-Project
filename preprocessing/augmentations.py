import os

import albumentations as A
import cv2

# desired transformations
rotate_90 = A.Compose([A.RandomRotate90(always_apply=True)])
rotate_180 = A.Compose([A.RandomRotate90(always_apply=True),
                        A.RandomRotate90(always_apply=True)])
rotate_270 = A.Compose([A.RandomRotate90(always_apply=True),
                        A.RandomRotate90(always_apply=True),
                        A.RandomRotate90(always_apply=True)])

flip_hor = A.Compose([A.HorizontalFlip(always_apply=True)])
flip_hor_90 = A.Compose([A.HorizontalFlip(always_apply=True),
                         A.RandomRotate90(always_apply=True)])
flip_ver = A.Compose([A.VerticalFlip(always_apply=True)])
flip_ver_90 = A.Compose([A.VerticalFlip(always_apply=True),
                         A.RandomRotate90(always_apply=True)])

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

            # apply the transformations
            # rotations
            print("...rotations")
            transform_and_save(image, mask, name="rotate_90", transform=rotate_90)
            transform_and_save(image, mask, name="rotate_180", transform=rotate_180)
            transform_and_save(image, mask, name="rotate_270", transform=rotate_270)

            # flipped image plus rotations
            print("...flipping and rotations")
            transform_and_save(image, mask, name="flip_hor", transform=flip_hor)
            transform_and_save(image, mask, name="flip_hor_90", transform=flip_hor_90)
            transform_and_save(image, mask, name="flip_ver", transform=flip_ver)  # flipped + 180
            transform_and_save(image, mask, name="flip_ver_90", transform=flip_ver_90)  # flipped + 270

            # random
            print("...random augmentations")
            transform_and_save(image, mask, name="crop_random", transform=crop_random)
            transform_and_save(image, mask, name="rotate_random", transform=rotate_random)
