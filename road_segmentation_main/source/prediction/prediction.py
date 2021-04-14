"""
Class for prediction of a set of images
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import os
import sys
import torch
import math
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image, ImageChops

from source.data.dataset import RoadSegmentationDatasetInference
from source.configuration import Configuration

class Prediction(object):

    def __init__(self, engine, images, device):
        self.device = device
        self.model = engine.model
        self.images_folder = images

    def get_crop_box(self, i, j, height, width, stride):
        """
         Determine crop box.
         :param i: x-axis index
         :param j: y-axis index
         :param height: total image height
         :param width: total image width
         :param stride: crop box size (x, y)
         :return: coordinates of top-left and bottom-right corner
        """
        # top left corner
        left = i * stride[0]  # x0
        upper = j * stride[1]  # y0
        # bottom right corner
        right = (i + 1) * stride[0]  # x1
        lower = (j + 1) * stride[1]  # y1

        if right > width:
            overlap = right - width
            right = width
            left -= overlap

        if lower > height:
            # shift box up
            overlap = lower - height
            lower = height
            upper -= overlap

        return left, upper, right, lower

    def get_cropped_images(self, image, stride=(400, 400)):
        height = image.height
        width = image.width

        cropped_images = []

        for i in range(math.ceil(height / stride[0])):
            for j in range(math.ceil(width / stride[1])):
                box = self.get_crop_box(i, j, height, width, stride)
                img_cropped = image.crop(box)

                # img_cropped.show()
                # from matplotlib import pyplot
                # pyplot.imshow(img_cropped)
                # pyplot.show()

                cropped_images.append(img_cropped)

        return cropped_images

    def patch_image_together(self, cropped_images, total_width=608, total_height=608, stride=(400, 400)):
        width = total_width
        height = total_height

        new_image = Image.new('RGB', (width, height))

        image_idx = 0
        for i in range(math.ceil(height / stride[0])):
            for j in range(math.ceil(width / stride[1])):
                left, upper, right, lower = self.get_crop_box(i, j, height, width, stride)

                new_image.paste(cropped_images[image_idx], (left, upper)) #Todo: check
                # new_image.show()
                image_idx += 1

        return new_image


    # assign a label to a patch
    def patch_to_label(self, patch, foreground_threshold):
        df = np.mean(patch)
        if df > foreground_threshold:
            return 1
        else:
            return 0

    def mask_to_submission_strings(self, preds, patch_size, foreground_threshold, image_nr):
        # iterate over prediction, just use every 16th pixel
        for i in range(0, preds.shape[0], patch_size):
            patch = preds[i:i + patch_size]
            label = self.patch_to_label(patch, foreground_threshold)
            yield ("{:03d}_{}_{},{}".format(image_nr, 1, i, label))


    def predict(self):

        image_list = os.listdir(self.images_folder)
        image_paths = [os.path.join(self.images_folder, image) for image in image_list]

        # Todo: add jonas code somewhere here

        dataset = RoadSegmentationDatasetInference(image_list=image_paths)
        loader = DataLoader(dataset, batch_size=8, num_workers=2, pin_memory=True, shuffle=False)

        patch_size = 16
        foreground_threshold = 0.5
        image_nr = 1222 #Todo: Find out image nr, maybe include in dataset

        with open(os.path.join(Configuration.output_directory, 'submission.csv'), 'w') as f:
            f.write('id,prediction\n')

            for idx, x in enumerate(loader):
                x = x.to(device=self.device)

                with torch.no_grad():
                    preds = self.model(x)

                    preds = torch.sigmoid(preds) # We need it because our models are constructend without sigmoid at the end
                    #split tensor into packages of 4 images and then
                    # for every package call patch_image_together and then f_write
                    # probabilities to 0/1
                    f.writelines('{}\n'.format(s) for s in self.mask_to_submission_strings(preds=preds, patch_size=patch_size, foreground_threshold=foreground_threshold, image_nr=image_nr))



