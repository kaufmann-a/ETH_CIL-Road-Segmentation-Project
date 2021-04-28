__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from source.helpers.utils import mask_to_submission_mask


class RoadSegmentationDataset(Dataset):
    def __init__(self, image_list, mask_list, threshold, transform=None, use_submission_masks=False):
        # self.device = device # unsure whether we need this, if yes: add parameter device to init
        self.transform = transform
        self.images = image_list
        self.masks = mask_list
        self.foreground_threshold = threshold
        self.use_submission_masks = use_submission_masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

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
    def __init__(self, image_list):
        # self.device = device # unsure whether we need this, if yes: add parameter device to init
        self.images = image_list

        # TODO get transformation from "central" place
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        self.transform = A.Compose(
            [
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        augmentations = self.transform(image=np.array(self.images[index]))

        return augmentations["image"]
