


__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class RoadSegmentationDataset(Dataset):
    def __init__(self, image_list, mask_list, device, transform=None):
        self.device = device
        self.transform = transform
        self.images = image_list
        self.masks = mask_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = self.images[index]
        mask_path = self.masks[index]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        threshold = 255.0 / 2
        mask = (mask >= threshold).astype(int)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
