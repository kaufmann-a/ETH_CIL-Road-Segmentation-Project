import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


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


class RoadSegmentationDatasetInference(Dataset):
    def __init__(self, image_list, transform):
        # self.device = device # unsure whether we need this, if yes: add parameter device to init
        self.images = image_list
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = np.array(self.images[index])
        return self.transform(img)


def get_mean_std(img_list):
    # https://stackoverflow.com/questions/60101240/finding-mean-and-standard-deviation-across-image-channels-pytorch

    dataset = RoadSegmentationDatasetInference(img_list, transform=transforms.Compose([transforms.ToTensor()]))
    loader = DataLoader(dataset, batch_size=1)

    """
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print("mean per channel:", mean)
    print("std per channel:", std)
    """

    nimages = 0
    mean = 0.0
    var = 0.0
    for batch in loader:
        # Rearrange batch to be the shape of [B, C, W * H]
        batch = batch.view(batch.size(0), batch.size(1), -1)
        # Update total number of images
        nimages += batch.size(0)
        # Compute mean and std here
        mean += batch.mean(2).sum(0)
        var += batch.var(2).sum(0)

    mean /= nimages
    var /= nimages
    std = torch.sqrt(var)

    print("mean per channel:", mean)
    print("std per channel:", std)


def is_valid_image(path):
    if ".png" in path and "images" in path:
        return True
    return False


if __name__ == '__main__':
    dirImages = "../../data/training/original/images/"
    dirGroundtruth = "../../data/training/original/masks"
    img_list, mask_list = read_images(dirImages, dirGroundtruth)
    display_images(img_list, mask_list, top=5)

    # compare mean and std
    print("TRAIN IMAGES")
    img_list = read_images(dirImages)
    get_mean_std(img_list)

    print("TEST IMAGES")
    dirTestImages = "../../data/test_images/"
    test_list = read_images(dirTestImages)
    get_mean_std(img_list)

    print("COMBINED")
    get_mean_std(img_list + test_list)
