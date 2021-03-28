import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class RoadSegmentationSimpleDataset(Dataset):
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

        # set values to 1 and 0 depending on the threshold
        threshold = 255.0 / 2
        mask = (mask >= threshold).astype(int)  # bool to 0, 1

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # return image.to(self.device), mask.to(self.device)
        return image, mask
