#!/usr/bin/env python3
# coding: utf8

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
from source.data.transformation import get_transformations
from source.helpers import predictionhelper, imagesavehelper
from source.helpers.image_cropping import get_crop_box, ImageCropper
from source.helpers.predictionhelper import run_post_processing
from source.logcreator.logcreator import Logcreator

PATCH_SIZE = 16

TEST_IMAGE_SIZE = (608, 608)


class Prediction(object):

    def __init__(self, engine, images, device, threshold, postprocessing, use_original_image_size,
                 enable_postprocessing=False,
                 use_submission_masks=False,
                 use_swa_model=False):
        """

        :param engine:
        :param images:
        :param device:
        :param threshold:
        :param use_original_image_size: False = patch image together from small subpatches of same size as in training
        """

        self.device = device
        self.model = engine.model
        self.swa_model = engine.swa_model
        self.model.to(device)
        self.images_folder = images
        self.foreground_threshold = threshold
        self.enable_postprocessing = enable_postprocessing
        self.postprocessing = postprocessing
        self.use_original_image_size = use_original_image_size
        self.use_submission_mask = use_submission_masks
        self.use_swa_model = use_swa_model

    def patch_image_together(self, cropped_images, mode='RGB', total_width=608, total_height=608, stride=(400, 400)):
        width = total_width
        height = total_height

        new_image = Image.new(mode, (width, height))

        image_idx = 0
        for i in range(math.ceil(height / stride[0])):
            for j in range(math.ceil(width / stride[1])):
                left, upper, right, lower = get_crop_box(i, j, height, width, stride)

                new_image.paste(cropped_images[image_idx], (left, upper))
                # new_image.show()
                image_idx += 1

        return new_image

    def patch_masks_together(self, cropped_images, out_image_size=(608, 608), stride=(400, 400),
                             mode='avg', debug=False):
        """
        Stitches a image mask together from multiple cropped image masks.

        :param cropped_images: list of 2d-image tensors
        :param out_image_size: (output image width, output image height)
        :param stride: size of cropped images
        :param mode: 'avg': take the average of the overlapping areas,
                     'max': take the maximum of the overlapping areas,
                     'overwrite': overwrite the overlapping areas
        :param debug: True = plot images

        :return: the combined image
        """
        width = out_image_size[0]
        height = out_image_size[1]

        out_array = torch.zeros(size=(width, height)).to(self.device)
        out_overlap_count = torch.ones(size=(width, height)).to(self.device)  # stores nr of intersection at pixels
        mem_right = mem_lower = right = lower = 0

        image_idx = 0
        for i in range(math.ceil(height / stride[0])):
            for j in range(math.ceil(width / stride[1])):
                mem_lower = lower if mem_lower < lower < height else mem_lower
                mem_right = right if mem_right < right < width else mem_right
                left, upper, right, lower = get_crop_box(i, j, height, width, stride)

                # TODO how to patch together: addition, average, smooth corners, ...?

                if mode == 'avg':
                    # get overlaping ranges
                    overlapping_indices_2 = torch.nonzero(out_array[upper:lower, left:right], as_tuple=True)
                    # Check if there is any overlap at all
                    if len(overlapping_indices_2[0]) > 0 and len(overlapping_indices_2[1]) > 0:
                        upper_overlap = int(torch.min(overlapping_indices_2[0])) + upper
                        lower_overlap = int(torch.max(overlapping_indices_2[0])) + upper + 1
                        left_overlap = int(torch.min(overlapping_indices_2[1])) + left
                        right_overlap = int(torch.max(overlapping_indices_2[1])) + left + 1

                        out_overlap_count[upper_overlap:lower_overlap, left_overlap:right_overlap] += 1

                        # case bottom right image was placed, we need to subtract bottom right square again
                        if torch.max(out_overlap_count) == 4:
                            out_overlap_count[mem_lower:lower, mem_right:right] -= 1

                    out_array[upper:lower, left:right] = torch.add(out_array[upper:lower, left:right],
                                                                   cropped_images[image_idx])
                elif mode == 'max':
                    out_array[upper:lower, left:right] = torch.maximum(out_array[upper:lower, left:right],
                                                                       cropped_images[image_idx])
                else:
                    # overwrite overlapping areas
                    out_array[upper:lower, left:right] = cropped_images[image_idx]

                if debug:
                    # print(left, upper, right, lower)
                    # print(cropped_images[image_idx].shape)
                    plot_array = out_array > 0.5
                    pyplot.imshow(plot_array.cpu(), cmap='gray', vmin=0, vmax=1)
                    pyplot.show()

                image_idx += 1
        if mode == 'avg':
            out_array = torch.div(out_array, out_overlap_count)
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

        image_cropper = ImageCropper(out_image_size=stride)

        for file in os.listdir(imgDir):
            filename = os.fsdecode(file)
            if filename.endswith(".png") or filename.endswith(".jpg"):
                # get image number
                image_number_list.append([int(s) for s in filename[:-4].split("_") if s.isdigit()][0])
                # get cropped images
                input_image = Image.open(os.path.join(imgDir, filename))
                cropped_images = image_cropper.get_cropped_images(input_image)

                # concatenate out-images with new cropped-images
                out_image_list += cropped_images

                if sanity_check:
                    img_patched = self.patch_image_together(cropped_images, mode='RGB', stride=stride)
                    if ImageChops.difference(input_image, img_patched).getbbox() is not None:
                        print("Images are not equal!")

        return out_image_list, image_number_list

    def run_prediction_loop(self, model, loader, cropped_image_size, nr_crops_per_image):
        mask_probabilistic_list = []

        model.eval()

        loop = tqdm(loader)
        for idx, x in enumerate(loop):
            x = x.to(device=self.device)

            with torch.no_grad():
                preds = model(x)

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
                    input_image_size = TEST_IMAGE_SIZE
                    out_patch_size = cropped_image_size
                    if self.use_submission_mask:
                        input_image_size = [x // PATCH_SIZE for x in input_image_size]
                        out_patch_size = [x // PATCH_SIZE for x in out_patch_size]
                    out_mask = self.patch_masks_together(cropped_images=crops_list,
                                                         out_image_size=input_image_size,
                                                         stride=out_patch_size)
                    mask_probabilistic_list.append(out_mask)

            loop.set_postfix(image_nr=len(mask_probabilistic_list))

        return mask_probabilistic_list

    def run_post_prediction_tasks(self, mask_probabilistic_list, image_number_list, path_prefix):
        Logcreator.info("Saving submission file")
        predictionhelper.images_to_submission_file(mask_probabilistic_list, image_number_list,
                                                   patch_size=PATCH_SIZE,
                                                   foreground_threshold=self.foreground_threshold,
                                                   folder=Configuration.output_directory,
                                                   file_prefix=path_prefix,
                                                   use_submission_mask=self.use_submission_mask)

        mask_binary_list = imagesavehelper.save_masks_as_images(mask_probabilistic_list, image_number_list,
                                                                folder=Configuration.output_directory,
                                                                is_prob=True,
                                                                pixel_threshold=self.foreground_threshold,
                                                                save_submission_img=not self.use_submission_mask,
                                                                folder_prefix=path_prefix)
        if self.enable_postprocessing:
            Logcreator.info("Running post processing")
            run_post_processing(mask_binary_list,
                                folder=Configuration.output_directory,
                                postprocessing_params=self.postprocessing,
                                image_number_list=image_number_list,
                                patch_size=PATCH_SIZE,
                                foreground_threshold=self.foreground_threshold,
                                path_prefix=path_prefix)

    def run_prediction(self, model, loader, image_number_list, cropped_image_size, nr_crops_per_image, file_prefix=''):
        mask_probabilistic_list = self.run_prediction_loop(model=model,
                                                           loader=loader,
                                                           cropped_image_size=cropped_image_size,
                                                           nr_crops_per_image=nr_crops_per_image)

        self.run_post_prediction_tasks(mask_probabilistic_list, image_number_list, file_prefix)

    def predict(self):
        if self.use_original_image_size:
            cropped_image_size = TEST_IMAGE_SIZE
        else:
            cropped_image_size = Configuration.get("training.general.cropped_image_size")

        image_list, image_number_list = self.load_test_images(self.images_folder, stride=cropped_image_size)
        nr_crops_per_image = int(len(image_list) / len(image_number_list))

        dataset = RoadSegmentationDatasetInference(image_list=image_list, transform=get_transformations())
        loader = DataLoader(dataset, batch_size=2 * nr_crops_per_image, num_workers=2, pin_memory=True, shuffle=False)

        self.run_prediction(self.model, loader, image_number_list, cropped_image_size, nr_crops_per_image)

        if self.use_swa_model:
            Logcreator.info("Stochastic Weight Averaging prediction run")
            self.run_prediction(self.swa_model, loader, image_number_list, cropped_image_size, nr_crops_per_image,
                                file_prefix='swa-')
