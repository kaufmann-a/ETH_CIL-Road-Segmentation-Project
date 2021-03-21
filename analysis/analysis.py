import os

import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':
    dirImages = "../data/training/images"
    dirGroundtruth = "../data/training/groundtruth"

    for file in os.listdir(dirImages):
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            img = Image.open(os.path.join(dirImages, filename))
            mask = Image.open(os.path.join(dirGroundtruth, filename))

            # make black transparent
            alpha = mask.getchannel(0)
            mask.putalpha(alpha)

            # plot
            DPI = 100.0
            plt.figure(figsize=(400 / DPI, 400 / DPI))
            plt.imshow(img)
            plt.imshow(mask, alpha=0.7)
            plt.show()
