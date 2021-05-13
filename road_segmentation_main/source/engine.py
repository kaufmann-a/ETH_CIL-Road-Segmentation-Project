#!/usr/bin/env python3
# coding: utf8

"""
Model of the road segmentatjion neuronal network learning object.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

from comet_ml import Experiment

import sys
import os
import numpy as np
import random
import torch

from torch.optim.swa_utils import AveragedModel
import torchmetrics
from torch.utils.data import DataLoader
from torchmetrics import IoU

from torchsummary import summary
from tqdm import tqdm
from io import StringIO

from source.configuration import Configuration
from source.data.datapreparator import DataPreparator
from source.helpers.imagesavehelper import save_predictions_as_imgs
import source.helpers.metricslogging as metricslogging
from source.logcreator.logcreator import Logcreator
from source.lossfunctions.lossfunctionfactory import LossFunctionFactory
from source.metrics.metrics import PatchAccuracy, GeneralAccuracyMetric
from source.models.modelfactory import ModelFactory
from source.optimizers.optimizerfactory import OptimizerFactory
from source.scheduler.lr_schedulerfactory import LRSchedulerFactory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Engine:

    def __init__(self):
        # fix random seeds
        seed = 49626446
        self.fix_random_seeds(seed)

        # initialize model
        self.model = ModelFactory.build().to(DEVICE)
        self.optimizer = OptimizerFactory.build(self.model)
        self.lr_scheduler = LRSchedulerFactory.build(self.optimizer)
        self.loss_function = LossFunctionFactory.build(self.model)
        self.scaler = torch.cuda.amp.GradScaler()  # I assumed we always use gradscaler, thus no factory for this
        self.submission_loss = Configuration.get("training.general.submission_loss")
        Logcreator.info("Following device will be used for training: " + DEVICE,
                        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

        # stochastic model weight averaging
        # https://pytorch.org/docs/stable/optim.html#constructing-averaged-models
        self.swa_model = AveragedModel(self.model).to(DEVICE)
        self.swa_enabled = Configuration.get("training.general.stochastic_weight_averaging.on")
        self.swa_start_epoch = Configuration.get("training.general.stochastic_weight_averaging.start_epoch")
        # TODO Maybe add learning rate scheduler for swa
        #  see https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        # self.swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, \
        #          anneal_strategy="linear", anneal_epochs=5, swa_lr=0.05)

        # Init comet and tensorboard
        self.experiment = metricslogging.init_comet()
        self.tensorboard = metricslogging.init_tensorboard()

        # Print model summary
        self.print_modelsummary()

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def train(self, epoch_nr=0):
        training_data, validation_data, test_data = DataPreparator.load()

        # Load training parameters from config file
        train_parms = Configuration.get('training.general')

        train_loader = DataLoader(training_data, batch_size=train_parms.batch_size, num_workers=train_parms.num_workers,
                                  pin_memory=True, shuffle=train_parms.shuffle_data)
        val_loader = DataLoader(validation_data, batch_size=train_parms.batch_size, num_workers=train_parms.num_workers,
                                pin_memory=True,
                                shuffle=False)  # TODO: check what shuffle exactly does and how to use it
        self.test_data = test_data

        epoch = 0
        if epoch_nr != 0:  # Check if continued training
            epoch = epoch_nr + 1  # plus one to continue with the next epoch

        while epoch < train_parms.num_epochs:
            Logcreator.info(f"Epoch {epoch}, lr: {self.get_lr():.3e}, lr-step: {self.lr_scheduler.last_epoch}")

            train_metrics = self.train_step(train_loader, epoch)
            val_metrics = self.evaluate(self.model, val_loader, epoch)

            # save model
            if (epoch % train_parms.checkpoint_save_interval == train_parms.checkpoint_save_interval - 1) or (
                    epoch + 1 == train_parms.num_epochs and DEVICE == "cuda"):
                self.save_model(epoch)
                self.save_checkpoint(self.model, epoch, train_metrics['train_loss'], train_metrics['train_acc'],
                                     val_metrics['val_loss'], val_metrics['val_acc'])

            # swa model
            if self.swa_enabled and epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                # update batch normalization statistics for the swa_model
                torch.optim.swa_utils.update_bn(train_loader, self.swa_model, device=DEVICE)
                # evaluate on validation set
                swa_val_metrics = self.evaluate(self.swa_model, val_loader, epoch,
                                                log_model_name="SWA-", log_postfix_path='val_swa')

                # save model
                if (epoch % train_parms.checkpoint_save_interval == train_parms.checkpoint_save_interval - 1) or (
                        epoch + 1 == train_parms.num_epochs and DEVICE == "cuda"):
                    if self.swa_enabled and epoch >= self.swa_start_epoch:
                        # update batch normalization statistics for the swa_model at the end
                        torch.optim.swa_utils.update_bn(train_loader, self.swa_model, device=DEVICE)
                        # save swa model separately
                        self.save_checkpoint(self.swa_model, epoch, train_metrics['train_loss'],
                                             train_metrics['train_acc'], swa_val_metrics['val_loss'],
                                             swa_val_metrics['val_acc'], file_name="swa_checkpoint.pth")

            # Flus tensorboard
            self.tensorboard.flush()

            epoch += 1

        # TODO: Maybe save the images also in tensorbaord log (every other epoch?)
        # save predicted validation images
        save_predictions_as_imgs(val_loader, self.model,
                                 folder=os.path.join(Configuration.output_directory, "prediction-validation-set"),
                                 device=DEVICE,
                                 is_prob=False,
                                 pixel_threshold=Configuration.get("inference.general.foreground_threshold"))

        return 0

    def train_step(self, data_loader, epoch):
        """
        Train model for 1 epoch.
        """
        self.model.train()

        # initialize metrics
        accuracy, iou, multi_accuracy_metric, patch_accuracy = self.get_metrics()

        total_loss = 0.

        # progressbar
        loop = tqdm(data_loader)

        # for all batches
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)

            self.optimizer.zero_grad()

            # runs the forward pass with autocasting (improve performance while maintaining accuracy)
            with torch.cuda.amp.autocast():
                predictions = self.model(data)
                loss = self.compute_loss(predictions, targets)

            # backward according to https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # update loss
            total_loss += loss.item()

            # update metrics
            # TODO create one Metrics class that we can feed with (predicted, targets)
            #  and computes all metrics we want and maybe logs them -> tensorboard?
            prediction_probabilities = torch.sigmoid(predictions)
            multi_accuracy_metric.update(prediction_probabilities, targets)
            accuracy.update(prediction_probabilities, targets.int())
            patch_accuracy.update(prediction_probabilities, targets)
            iou.update(prediction_probabilities, targets.int())

            # update tqdm progressbar
            loop.set_postfix(loss=loss.item())

        self.lr_scheduler.step()  # decay learning rate over time

        # compute epoch scores
        train_loss = total_loss / len(data_loader)
        train_acc = accuracy.compute()
        train_patch_acc = patch_accuracy.compute()
        train_iou_score = iou.compute()

        # log scores
        multi_accuracy_metric.compute_and_log(self.tensorboard, epoch, path_postfix='train')
        # Tensorboard
        self.tensorboard.add_scalar("Loss/train", train_loss, epoch)
        self.tensorboard.add_scalar("Accuracy/train", train_acc, epoch)
        self.tensorboard.add_scalar("PatchAccuracy/train", train_patch_acc, epoch)
        self.tensorboard.add_scalar("IoU/train", train_iou_score, epoch)
        # Comet
        if self.experiment is not None:
            self.experiment.log_metric('train_loss', train_loss)
            self.experiment.log_metric('train_acc', train_acc)
            self.experiment.log_metric('train_patch_acc', train_patch_acc)
            self.experiment.log_metric('train_iou_score', train_iou_score)
        # Logfile
        Logcreator.info(f"Training:   loss: {train_loss:.5f}",
                        f", accuracy: {train_acc:.5f}",
                        f", patch-acc: {train_patch_acc:.5f}",
                        f", IoU: {train_iou_score:.5f}")

        return {'train_loss': total_loss, 'train_acc': accuracy.compute(), 'train_patch_acc': patch_accuracy.compute(),
                'train_iou_score': train_iou_score}

    def evaluate(self, model, data_loader, epoch, log_model_name='', log_postfix_path='val'):
        """
        Evaluate model on validation data.
        """
        model.eval()

        # initialize metrics
        accuracy, iou, multi_accuracy_metric, patch_accuracy = self.get_metrics()

        total_loss = 0.

        with torch.no_grad():
            for i, (image, targets) in enumerate(data_loader):
                image, targets = image.to(DEVICE), targets.to(DEVICE)

                # forward pass according to https://pytorch.org/docs/stable/amp.html
                with torch.cuda.amp.autocast():
                    predictions = model(image)
                    loss = self.compute_loss(predictions, targets)

                # update loss
                total_loss += loss.item()

                # update metrics
                prediction_probabilities = torch.sigmoid(predictions)
                multi_accuracy_metric.update(prediction_probabilities, targets)
                accuracy.update(prediction_probabilities, targets.int())
                patch_accuracy.update(prediction_probabilities, targets)
                iou.update(prediction_probabilities, targets.int())

        # compute epoch scores
        val_loss = total_loss / len(data_loader)
        val_acc = accuracy.compute()
        val_patch_acc = patch_accuracy.compute()
        val_iou_score = iou.compute()

        # log scores
        multi_accuracy_metric.compute_and_log(self.tensorboard, epoch, path_postfix=log_postfix_path)
        # Tensorboard
        self.tensorboard.add_scalar("Loss/" + log_postfix_path, val_loss, epoch)
        self.tensorboard.add_scalar("Accuracy/" + log_postfix_path, val_acc, epoch)
        self.tensorboard.add_scalar("PatchAccuracy/" + log_postfix_path, val_patch_acc, epoch)
        self.tensorboard.add_scalar("IoU/val", val_iou_score, epoch)
        # Comet
        if self.experiment is not None:
            self.experiment.log_metric(log_postfix_path + '_loss', val_loss)
            self.experiment.log_metric(log_postfix_path + '_acc', val_acc)
            self.experiment.log_metric(log_postfix_path + '_patch_acc', val_patch_acc)
            self.experiment.log_metric('val_iou_score', val_iou_score)
        # Logfile
        Logcreator.info(log_model_name + f"Validation: loss: {val_loss:.5f}",
                        f", accuracy: {val_acc:.5f}",
                        f", patch-acc: {val_patch_acc:.5f}",
                        f", IoU: {val_iou_score:.5f}")

        return {'val_loss': total_loss, 'val_acc': val_acc, 'val_patch_acc': val_patch_acc,
                'val_iou_score': val_iou_score}

    def get_metrics(self):
        multi_accuracy_metric = GeneralAccuracyMetric(device=DEVICE)
        foreground_threshold = Configuration.get('training.general.foreground_threshold')
        accuracy = torchmetrics.Accuracy(threshold=foreground_threshold)
        accuracy.to(DEVICE)
        patch_accuracy = PatchAccuracy(threshold=foreground_threshold)
        patch_accuracy.to(DEVICE)
        iou = IoU(num_classes=2, threshold=foreground_threshold)
        iou.to(DEVICE)
        return accuracy, iou, multi_accuracy_metric, patch_accuracy

    def save_model(self, epoch_nr):
        """ This function saves entire model incl. modelstructure"""
        Configuration.model_save_folder = os.path.join(Configuration.output_directory, "whole_model_backups")
        if not os.path.exists(Configuration.model_save_folder):
            os.makedirs(Configuration.model_save_folder)
        file_name = str(epoch_nr) + "_whole_model_serialized.pth"
        torch.save(self.model, os.path.join(Configuration.model_save_folder, file_name))

    def save_checkpoint(self, model, epoch, tl, ta, vl, va, file_name="checkpoint.pth"):
        Configuration.weights_save_folder = os.path.join(Configuration.output_directory, "weights_checkpoint")
        if not os.path.exists(Configuration.weights_save_folder):
            os.makedirs(Configuration.weights_save_folder)
        file_name = str(epoch) + "_" + file_name
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': tl,
            'train_accuracy': ta,
            'val_loss': vl,
            'val_accuracy': va,
        }, os.path.join(Configuration.weights_save_folder, file_name))

    def load_model(self, path=None):
        self.model = torch.load(path)
        self.model.eval()  # Todo: check if needed

    def load_checkpoints(self, path=None, reset_lr=False):
        """
        Loads a checkpoint.

        :param path: path to checkpoint file.
        :param reset_lr: True = resets the learning rate of the optimizer to the configuration values.
        :return:
        """
        Logcreator.info("Loading checkpoint file:", path)

        checkpoint = torch.load(path)
        # Todo: check if to device should be called somewhere
        self.model.load_state_dict(checkpoint['model_state_dict'])

        initial_lr = self.get_lr()
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if reset_lr:
            self.optimizer.param_groups[0]["lr"] = initial_lr
            Logcreator.info(f"Reseted learning rate to: {self.get_lr():e}")

        epoch = checkpoint['epoch']

        # TODO should we do this or should we start from zero or save the scheduler?
        # init the step count of the learning rate scheduler
        self.lr_scheduler = LRSchedulerFactory.build(self.optimizer, last_epoch=epoch + 1)

        train_loss = checkpoint['train_loss']
        train_accuracy = checkpoint['train_accuracy']
        val_loss = checkpoint['val_loss']
        val_accuracy = checkpoint['val_accuracy']
        # ToDo: add patch_accuracy to checkpoints

        return epoch, train_loss, train_accuracy, val_loss, val_accuracy

    def compute_loss(self, predictions, targets):
        if self.submission_loss:
            avgPool = torch.nn.AvgPool2d(16, stride=16)
            predictions = avgPool(predictions)
            targets = avgPool(targets)

        return self.loss_function(predictions, targets)

    def fix_random_seeds(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def print_modelsummary(self):
        cropped_image_size = Configuration.get("training.general.cropped_image_size")
        input_size = tuple(np.insert(cropped_image_size, 0, values=3))
        # redirect stdout to our logger
        sys.stdout = mystdout = StringIO()
        summary(self.model, input_size=input_size, device=DEVICE)
        # reset stdout to original
        sys.stdout = sys.__stdout__
        Logcreator.info(mystdout.getvalue())

        Logcreator.debug("Model '%s' initialized with %d parameters." %
                         (Configuration.get('training.model.name'),
                          sum(p.numel() for p in self.model.parameters() if p.requires_grad)))
