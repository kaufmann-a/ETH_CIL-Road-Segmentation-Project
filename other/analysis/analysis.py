import os

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

if __name__ == '__main__':
    dirImages = "../../data/training/original/images"
    dirGroundtruth = "../../data/training/original/masks"

    for file in os.listdir(dirImages)[0:20]:
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            img = Image.open(os.path.join(dirImages, filename))
            mask = Image.open(os.path.join(dirGroundtruth, filename))

            # make black transparent
            alpha = mask.getchannel(0)
            mask.putalpha(alpha)

            height, width = img.size

            # plot
            DPI = 100.0
            fig = plt.figure(figsize=(400 / DPI, 400 / DPI))
            ax = fig.add_subplot(1, 1, 1)
            plt.imshow(img)
            plt.imshow(mask, alpha=0.7)

            grid_x_ticks = np.arange(0, width, 16)
            grid_y_ticks = np.arange(0, height, 16)
            ax.set_xticks(grid_x_ticks , minor=True)
            ax.set_yticks(grid_y_ticks , minor=True)

            plt.grid(color="red", which="minor")


            plt.show()
