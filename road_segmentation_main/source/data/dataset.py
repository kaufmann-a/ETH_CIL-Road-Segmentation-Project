__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from source.helpers.image_cropping import ImageCropper
from source.helpers.maskconverthelper import mask_to_submission_mask


class RoadSegmentationDataset(Dataset):
    def __init__(self, image_list, mask_list, threshold, transform=None,
                 crop_size=(400, 400),
                 use_submission_masks=False,
                 preload_images=True):
        # self.device = device # unsure whether we need this, if yes: add parameter device to init
        self.transform = transform
        self.images = image_list
        self.masks = mask_list
        self.foreground_threshold = threshold
        self.use_submission_masks = use_submission_masks
        self.preload_images = preload_images

        # open one image to get the size of the image
        self.image_cropper = ImageCropper(out_image_size=crop_size)
        if len(image_list) > 0:
            image = Image.open(self.images[0])
            self.cropping_needed = image.width > crop_size[0] or image.height > crop_size[1]
            self.nr_segments_per_image = self.image_cropper.get_number_of_cropped_images(image)
        else:
            self.cropping_needed = False
            self.nr_segments_per_image = 1

        # preload images into memory to not read from drive everytime
        self.images_preloaded = list()
        self.masks_preloaded = list()

        if self.preload_images:
            for image_path, mask_path in zip(self.images, self.masks):
                image = Image.open(image_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")

                if not self.cropping_needed:
                    # no cropping necessary
                    image = np.array(image)
                    mask = np.array(mask, dtype=np.float32)
                    # append image and mask to list
                    self.images_preloaded.append(image)
                    self.masks_preloaded.append(mask)
                else:
                    # get cropped images and masks
                    images_cropped = self.image_cropper.get_cropped_images(image)
                    masks_cropped = self.image_cropper.get_cropped_images(mask)

                    images = [np.array(_img) for _img in images_cropped]
                    masks = [np.array(_mask) for _mask in masks_cropped]

                    # concatenate lists
                    self.images_preloaded += images
                    self.masks_preloaded += masks

            # set to one since we preload all sub-patches
            self.nr_segments_per_image = 1

    def __len__(self):
        if self.preload_images:
            return len(self.images_preloaded)
        else:
            return len(self.images) * self.nr_segments_per_image

    def __getitem__(self, index):
        index_into_list = index // self.nr_segments_per_image
        index_of_segment = index % self.nr_segments_per_image

        if self.preload_images:
            image = self.images_preloaded[index_into_list]
            mask = self.masks_preloaded[index_into_list]
        else:
            img_path = self.images[index_into_list]
            mask_path = self.masks[index_into_list]

            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")

            if not self.cropping_needed:
                # no cropping necessary
                image = np.array(image)
                mask = np.array(mask, dtype=np.float32)
            else:
                # get cropped image and mask
                image_cropped = self.image_cropper.get_cropped_image(image, index_of_segment)
                mask_cropped = self.image_cropper.get_cropped_image(mask, index_of_segment)

                # sanity check in case no cropping is necessary
                # if self.nr_segments_per_image == 1:
                #    assert (np.equal(np.array(image), np.array(image_cropped)).all())
                #    assert (np.equal(np.array(mask), np.array(mask_cropped)).all())

                image = np.array(image_cropped)
                mask = np.array(mask_cropped, dtype=np.float32)

        threshold = 255.0 * self.foreground_threshold
        mask = (mask >= threshold).astype(int)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

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
