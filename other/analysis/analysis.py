import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def display_images(img_list, mask_list, top=10):
    for idx, [img, mask] in enumerate(zip(img_list, mask_list)):
        # make black transparent
        alpha = mask.getchannel(0)
        mask.putalpha(alpha)

        height, width = img.size

        # plot
        DPI = 100.0
        fig = plt.figure(figsize=(height / DPI, width / DPI))
        ax = fig.add_subplot(1, 1, 1)
        plt.imshow(img)
        plt.imshow(mask, alpha=0.7)

        grid_x_ticks = np.arange(0, width, 16)
        grid_y_ticks = np.arange(0, height, 16)
        ax.set_xticks(grid_x_ticks, minor=True)
        ax.set_yticks(grid_y_ticks, minor=True)

        plt.grid(color="red", which="minor")

        plt.show()

        if idx == top:
            break


def read_images(dirImages, dirGroundtruth=None):
    img_list = list()
    mask_list = list()
    for file in os.listdir(dirImages)[0:20]:
        filename = os.fsdecode(file)
        if filename.endswith(".png"):
            img = Image.open(os.path.join(dirImages, filename))
            img_list.append(img)

            if dirGroundtruth is not None:
                mask = Image.open(os.path.join(dirGroundtruth, filename))
                mask_list.append(mask)

    if dirGroundtruth is None:
        return img_list

    return img_list, mask_list


def is_valid_image(path):
    if ".png" in path and "images" in path:
        return True
    return False


if __name__ == '__main__':
    dirImages = "../../data/training/eth_dataset/original/images/"
    dirGroundtruth = "../../data/training/eth_dataset/original/masks"
    img_list, mask_list = read_images(dirImages, dirGroundtruth)
    display_images(img_list, mask_list, top=5)
