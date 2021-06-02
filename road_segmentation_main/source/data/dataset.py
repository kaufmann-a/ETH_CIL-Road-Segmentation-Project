__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from source.helpers.image_cropping import ImageCropper
from source.helpers.maskconverthelper import mask_to_submission_mask
from source.logcreator.logcreator import Logcreator


class RoadSegmentationDataset(Dataset):
    def __init__(self, image_list, mask_list, threshold, transform=None,
                 crop_size=(400, 400),
                 use_submission_masks=False,
                 min_road_percentage=0.0001,
                 debug=False):
        # self.device = device # unsure whether we need this, if yes: add parameter device to init
        self.transform = transform
        self.images = image_list
        self.masks = mask_list
        self.foreground_threshold = threshold
        self.use_submission_masks = use_submission_masks
        self.min_road_percentage = min_road_percentage if 1 >= min_road_percentage >= 0 else 0
        self.image_cropper = ImageCropper(out_image_size=crop_size)
        # preload images into memory to not read from drive everytime
        self.images_preloaded = list()
        self.masks_preloaded = list()

        Logcreator.info("Removing images that contain less than,",
                        crop_size[0] * crop_size[1] * self.min_road_percentage,
                        f"road pixels (less than {self.min_road_percentage}% of the image is road)")

        count_removed_images = 0

        loop = tqdm(zip(self.images, self.masks), total=len(self.images), file=sys.stdout, desc="Preload images")
        for image_path, mask_path in loop:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            # get cropped images and masks
            images_cropped = self.image_cropper.get_cropped_images(image)
            masks_cropped = self.image_cropper.get_cropped_images(mask)

            images = [np.array(_img) for _img in images_cropped]
            masks = [np.array(_mask) for _mask in masks_cropped]

            if self.min_road_percentage > 0:
                # remove images with less percentage of road the given threshold
                keep_idx = [(np.sum(_mask > 0) / np.size(_mask) >= self.min_road_percentage) for _mask in masks]
                count_removed_images += np.sum(np.asarray(keep_idx) == False)

                if debug:
                    for _img, _mask, keep in zip(images, masks, keep_idx):
                        if keep == False:
                            plt.imshow(_img)
                            plt.show()
                            plt.imshow(_mask)
                            plt.show()

                images = [_img for _img, keep in zip(images, keep_idx) if keep]
                masks = [_mask for _mask, keep in zip(masks, keep_idx) if keep]

            # concatenate lists
            self.images_preloaded += images
            self.masks_preloaded += masks

            # set to one since we preload all sub-patches
            self.nr_segments_per_image = 1

        Logcreator.info("Nr. of cropped image in total:", str(len(self.images_preloaded)))
        Logcreator.info("Nr. of removed images:", str(count_removed_images))

    def __len__(self):

        return len(self.images_preloaded)

    def __getitem__(self, index):
        image = self.images_preloaded[index]
        mask = self.masks_preloaded[index]

        threshold = 255.0 * self.foreground_threshold
        mask = (mask >= threshold).astype(int)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

            # 2d to 3d tensor
            mask = mask.unsqueeze(0).float()

            if self.use_submission_masks:  # use submission masks for training
                mask = mask_to_submission_mask(mask, threshold=self.foreground_threshold)

        return image, mask


class RoadSegmentationDatasetInference(Dataset):
    def __init__(self, image_list, transform):
        # self.device = device # unsure whether we need this, if yes: add parameter device to init
        self.images = image_list
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        augmentations = self.transform(image=np.array(self.images[index]))

        return augmentations["image"]


class SimpleToTensorDataset(Dataset):
    def __init__(self, image_path_list):
        self.img_path_list = image_path_list
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img = Image.open(self.img_path_list[index]).convert("RGB")
        img = np.array(img)
        return self.transform(img)
