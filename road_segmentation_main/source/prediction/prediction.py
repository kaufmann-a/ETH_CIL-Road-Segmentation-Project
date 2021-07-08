#!/usr/bin/env python3
# coding: utf8

"""
Class for prediction of a set of images
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from source.configuration import Configuration
from source.data.datapreparator import DataPreparator
from source.helpers import predictionhelper, imagesavehelper
from source.helpers.image_cropping import ImageCropper
from source.helpers.predictionhelper import run_post_processing
from source.logcreator.logcreator import Logcreator

PATCH_SIZE = 16

TEST_IMAGE_SIZE = (608, 608)


class Prediction(object):

    def __init__(self, engine, image_folder, device, threshold, postprocessing_params, use_original_image_size,
                 enable_postprocessing=False,
                 use_submission_masks=False,
                 use_swa_model=False):
        """

        :param engine: The main engine.
        :param image_folder:  The path to the image folder.
        :param device: cuda or cpu
        :param threshold: The threshold used to convert 16x16 image patches to road labels.
        :param postprocessing_params: Object with the postprocessing parameters.
        :param use_original_image_size: False = Patch image together from small subpatches of same size as in training.
        :param enable_postprocessing: True = Postprocessing is executed.
        :param use_submission_masks: True = No 16x16 converting needs to be done, 
                                            since the model predicts already 16x16 patches.
        :param use_swa_model: True = use the swa model to predict road masks. 
        """

        self.device = device
        self.engine = engine
        self.model = engine.model
        self.swa_model = engine.swa_model
        self.model.to(device)
        self.images_folder = image_folder
        self.foreground_threshold = threshold
        self.enable_postprocessing = enable_postprocessing
        self.postprocessing_params = postprocessing_params
        self.use_original_image_size = use_original_image_size
        self.use_submission_mask = use_submission_masks
        self.use_swa_model = use_swa_model

    def patch_predictions_together(self, preds, cropped_image_size, nr_crops_per_image):
        """
        Patches the target image together from smaller sub-images if necessary.

        :param preds: The model predictions.
        :param cropped_image_size: The sub-images size.
        :param nr_crops_per_image: The number of sub-images per target image.
        :return: predictions in original image size
        """
        mask_probabilistic_list = []

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
            out_mask = ImageCropper.patch_masks_together(cropped_images=crops_list,
                                                         device=self.device,
                                                         out_image_size=input_image_size,
                                                         stride=out_patch_size)
            mask_probabilistic_list.append(out_mask)

        return mask_probabilistic_list

    def model_prediction_run(self, model, loader):
        """
        Makes prediction for all images in the data loader and returns them.

        :param loader: the data loader
        :param model: the model used to predict
        :return: predictions
        """
        preds_masks_list = []

        model.eval()

        loop = tqdm(loader, file=sys.stdout, desc="Prediction run")
        for idx, data in enumerate(loop):
            if isinstance(data, list):  # if data is a list it contains the ground truth
                images, masks = data
            else:
                images = data

            images = images.to(device=self.device)

            with torch.no_grad():
                preds = model(images)

                # get probabilities
                preds = torch.sigmoid(preds)

                preds_masks_list.append(preds)

            loop.set_postfix(image_nr=len(preds_masks_list))

        # convert list to 4d tensor
        preds_masks_list = torch.vstack(preds_masks_list)

        return preds_masks_list

    def run_post_prediction(self, mask_probabilistic_list, image_number_list, path_prefix):
        """
        Runs post prediction tasks like saving the submission file and running postprocessing.

        :param mask_probabilistic_list: list containing the predicted road masks
        :param image_number_list: list containing the corresponding image number
        :param path_prefix: a prefix for the folder name
        """
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
                                postprocessing_params=self.postprocessing_params,
                                image_number_list=image_number_list,
                                patch_size=PATCH_SIZE,
                                foreground_threshold=self.foreground_threshold,
                                path_prefix=path_prefix)

    def prediction_loop(self, model, loader, file_prefix=''):
        """
        Runs one prediction loop for with the provided model and data loader.

        :param model: The model used to predict masks.
        :param loader: The data loader.
        :param file_prefix: A prefix for the folder/file name.
        """
        # get predictions
        preds = self.model_prediction_run(model, loader)

        # patch predictions together to target image size
        cropped_image_size = loader.dataset.crop_size
        nr_crops_per_image = loader.dataset.nr_crops_per_image
        mask_probabilistic_list = self.patch_predictions_together(preds,
                                                                  cropped_image_size=cropped_image_size,
                                                                  nr_crops_per_image=nr_crops_per_image)
        # run postprocessing
        img_nr_list = loader.dataset.image_number_list
        self.run_post_prediction(mask_probabilistic_list, img_nr_list, file_prefix)

    def predict_test_images(self):
        """
        Executes the predictions on the test set and creates the submission file.
        """
        if self.use_original_image_size:
            cropped_image_size = TEST_IMAGE_SIZE
        else:
            cropped_image_size = Configuration.get("training.general.cropped_image_size")

        dataset = DataPreparator.load_test(self.engine, path=self.images_folder, crop_size=cropped_image_size)

        loader = DataLoader(dataset, batch_size=4, num_workers=2, pin_memory=True, shuffle=False)

        self.prediction_loop(self.model, loader)

        if self.use_swa_model:
            Logcreator.info("Stochastic Weight Averaging prediction run")
            self.prediction_loop(self.swa_model, loader, file_prefix='swa-')

    def predict_train_images(self, create_collection_folder_structure=False):
        """
        Executes the prediction on the entire training dataset.

        :param create_collection_folder_structure: True = Saves the predictions with the same folder structure as
               the images in the collection folders including three parent folders.
        """
        ds = DataPreparator.load_all(self.engine, is_train=False)
        loader = DataLoader(ds, batch_size=4, num_workers=2, pin_memory=True, shuffle=False)

        preds_masks_list = self.model_prediction_run(self.model, loader)

        from pathlib import Path
        file_paths = [Path(img) for img in ds.images_filtered]

        folder_out_path = Path(os.path.join(Configuration.output_directory, "pred-masks-original"))
        folder_out_path.mkdir(parents=True, exist_ok=True)  # create folders if they do not exist

        import torchvision

        for prediction_mask, file_path in tqdm(zip(preds_masks_list, file_paths),
                                               total=len(file_paths),
                                               file=sys.stdout,
                                               desc="Saving preds."):
            # probabilities to binary
            out_preds = (prediction_mask > self.foreground_threshold).float()

            # create folder structure
            if create_collection_folder_structure:
                file_relative_path = file_path.relative_to(file_path.parent.parent.parent.parent)
                file_out_path = folder_out_path.joinpath(file_relative_path)
            else:
                file_out_path = folder_out_path.joinpath(file_path.name)

            file_out_path.parent.mkdir(parents=True, exist_ok=True)

            # save prediction
            torchvision.utils.save_image(out_preds, file_out_path.absolute())
