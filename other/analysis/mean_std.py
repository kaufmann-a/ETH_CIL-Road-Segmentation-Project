import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from other.analysis.analysis import read_images


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


if __name__ == '__main__':
    dirImages = "../../data/training/eth_dataset/original/images/"
    dirGroundtruth = "../../data/training/original/masks"
    dirTestImages = "../../data/test_images/"

    # compare mean and std
    print("TRAIN IMAGES")
    img_list = read_images(dirImages)
    get_mean_std(img_list)

    print("TEST IMAGES")

    test_list = read_images(dirTestImages)
    get_mean_std(test_list)

    print("COMBINED")
    get_mean_std(img_list + test_list)
