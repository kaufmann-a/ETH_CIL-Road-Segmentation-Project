"""
Class for prediction of a set of images
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import math
import os

import torch
from PIL import Image, ImageChops
from matplotlib import pyplot
from torch.utils.data import DataLoader
from tqdm import tqdm

from source.configuration import Configuration
from source.data.dataset import RoadSegmentationDatasetInference


class Prediction(object):

    def __init__(self, engine, images, device):
        self.device = device
        self.model = engine.model
        self.model.to(device)
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

    def patch_image_together(self, cropped_images, mode='RGB', total_width=608, total_height=608, stride=(400, 400)):
        width = total_width
        height = total_height

        new_image = Image.new(mode, (width, height))

        image_idx = 0
        for i in range(math.ceil(height / stride[0])):
            for j in range(math.ceil(width / stride[1])):
                left, upper, right, lower = self.get_crop_box(i, j, height, width, stride)

                new_image.paste(cropped_images[image_idx], (left, upper))
                # new_image.show()
                image_idx += 1

        return new_image

    def patch_masks_together(self, cropped_images, total_width=608, total_height=608, stride=(400, 400), debug=False):
        width = total_width
        height = total_height

        out_array = torch.zeros(size=(width, height)).to(self.device)

        image_idx = 0
        for i in range(math.ceil(height / stride[0])):
            for j in range(math.ceil(width / stride[1])):
                left, upper, right, lower = self.get_crop_box(i, j, height, width, stride)

                # TODO how to patch together: addition, average, smooth corners, ...?
                out_array[upper:lower, left:right] = torch.add(out_array[upper:lower, left:right],
                                                               cropped_images[image_idx])
                out_array[upper:lower, left:right] /= 2

                if debug:
                    # print(left, upper, right, lower)
                    # print(cropped_images[image_idx].shape)
                    plot_array = out_array > 0.5
                    pyplot.imshow(plot_array.cpu(), cmap='gray', vmin=0, vmax=1)
                    pyplot.show()

                image_idx += 1

        return out_array

    def load_test_images(self, imgDir='../data/test_images/', stride=(400, 400), sanity_check=False):
        """
        Loads images from a directory and creates multiple cropped images (that if patched together
         again equal the original image) according to the stride.

        :param imgDir:
        :param stride: size of cropped output images (width, height)
        :param sanity_check: check if cropped images form again the original image

        :return: list of cropped images, image numbers
        """
        out_image_list = []
        image_number_list = []

        for file in os.listdir(imgDir):
            filename = os.fsdecode(file)
            if filename.endswith(".png"):
                # get image number
                image_number_list.append([int(s) for s in filename.removesuffix(".png").split("_") if s.isdigit()][0])
                # get cropped images
                input_image = Image.open(os.path.join(imgDir, filename))
                cropped_images = self.get_cropped_images(input_image, stride=stride)

                # concatenate out-images with new cropped-images
                out_image_list += cropped_images

                if sanity_check:
                    img_patched = self.patch_image_together(cropped_images, mode='RGB', stride=stride)
                    if ImageChops.difference(input_image, img_patched).getbbox() is not None:
                        print("Images are not equal!")

        return out_image_list, image_number_list

    def patch_to_label(self, patch, foreground_threshold):
        # assign a label to a patch
        df = torch.mean(patch)
        if df > foreground_threshold:
            return 1
        else:
            return 0

    def mask_to_submission_strings(self, image, image_nr, patch_size=16, foreground_threshold=0.5):
        # iterate over prediction, just use every 16th pixel
        for j in range(0, image.shape[1], patch_size):
            for i in range(0, image.shape[0], patch_size):
                patch = image[i:i + patch_size, j:j + patch_size]
                label = self.patch_to_label(patch, foreground_threshold)
                yield ("{:03d}_{}_{},{}".format(image_nr, j, i, label))

    def predict(self):
        image_list, image_number_list = self.load_test_images(self.images_folder)
        nr_crops_per_image = int(len(image_list) / len(image_number_list))

        dataset = RoadSegmentationDatasetInference(image_list=image_list)
        loader = DataLoader(dataset, batch_size=2 * nr_crops_per_image, num_workers=2, pin_memory=True, shuffle=False)

        patch_size = 16
        foreground_threshold = 0.5

        with open(os.path.join(Configuration.output_directory, 'submission.csv'), 'w') as f:
            f.write('id,prediction\n')

            self.model.eval()

            loop = tqdm(loader)
            image_nr_list_idx = 0
            for idx, x in enumerate(loop):
                x = x.to(device=self.device)

                with torch.no_grad():
                    preds = self.model(x)

                    # We need it because our models are constructed without sigmoid at the end
                    preds = torch.sigmoid(preds)

                    # go through all images of current batch; an image consists of multiple cropped images
                    for i in range(preds.shape[0] // nr_crops_per_image):
                        # split tensor into packages of "nr_crops_per_image" crops and then
                        pred_masks = preds[i * nr_crops_per_image:i * nr_crops_per_image + nr_crops_per_image]
                        crops_list = []
                        for j in range(nr_crops_per_image):
                            crops_list.append(torch.squeeze(pred_masks[j]))

                        # for every package call patch_image_together to get the original size image
                        out_image = self.patch_masks_together(cropped_images=crops_list)

                        # and then convert mask to string
                        f.writelines('{}\n'.format(s)
                                     for s in self.mask_to_submission_strings(image=out_image,
                                                                              patch_size=patch_size,
                                                                              foreground_threshold=foreground_threshold,
                                                                              image_nr=image_number_list[
                                                                                  image_nr_list_idx]))
                        image_nr_list_idx += 1

                loop.set_postfix(image_nr=image_nr_list_idx)
