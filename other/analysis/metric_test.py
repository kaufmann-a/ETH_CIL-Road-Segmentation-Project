import torch

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

if __name__ == '__main__':
    dirImages = "../../data/training/original/images"
    dirGroundtruth = "../../data/training/original/masks"

    for file in os.listdir(dirImages)[0:5]:
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            img = Image.open(os.path.join(dirImages, filename))
            mask = Image.open(os.path.join(dirGroundtruth, filename))

            # make black transparent
            alpha = mask.getchannel(0)
            mask.putalpha(alpha)

            height, width = img.size

            mask = np.array(mask.convert("L"), dtype=np.float32)

            mask_tensor = torch.Tensor(mask)
            mask_tensor = torch.unsqueeze(mask_tensor, dim=0)
            mask_tensor = torch.unsqueeze(mask_tensor, dim=0)

            avgPool = torch.nn.AvgPool2d(16, stride=16)

            patched_mask = avgPool(mask_tensor)
            patched_binary = (patched_mask >= 0.25).int()

            plt.subplot(1, 3, 1)
            plt.imshow(mask)
            plt.subplot(1, 3, 2)
            plt.imshow(torch.squeeze(patched_mask).numpy())
            plt.subplot(1,3,3)
            plt.imshow(torch.squeeze(patched_binary).numpy())
            plt.show()