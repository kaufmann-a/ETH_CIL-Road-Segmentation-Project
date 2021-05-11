#!/usr/bin/env python3
# coding: utf8

"""
Model of the road segmentatjion neuronal network learning object.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import os

import numpy as np
import torch
import torchmetrics as torchmetrics
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

from source.configuration import Configuration
from source.data.datapreparator import DataPreparator
from source.helpers.imagesavehelper import save_predictions_as_imgs
from source.logcreator.logcreator import Logcreator
from source.lossfunctions.lossfunctionfactory import LossFunctionFactory
from source.metrics.metrics import PatchAccuracy, GeneralAccuracyMetric
from source.models.modelfactory import ModelFactory
from source.optimizers.optimizerfactory import OptimizerFactory
from source.scheduler.lr_schedulerfactory import LRSchedulerFactory

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Engine:

    def __init__(self):
        self.model = ModelFactory.build().to(DEVICE)
        self.optimizer = OptimizerFactory.build(self.model)
        self.lr_scheduler = LRSchedulerFactory.build(self.optimizer)
        self.loss_function = LossFunctionFactory.build(self.model)
        self.scaler = torch.cuda.amp.GradScaler()  # I assumed we always use gradscaler, thus no factory for this
        self.submission_loss = Configuration.get("training.general.submission_loss")
        # stochastic model weight averaging
        # https://pytorch.org/docs/stable/optim.html#constructing-averaged-models
        self.swa_model = AveragedModel(self.model).to(DEVICE)
        self.swa_enabled = Configuration.get("training.general.stochastic_weight_averaging.on")
        self.swa_start_epoch = Configuration.get("training.general.stochastic_weight_averaging.start_epoch")
        # TODO Maybe add learning rate scheduler for swa
        #  see https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/
        # self.swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, \
        #          anneal_strategy="linear", anneal_epochs=5, swa_lr=0.05)

        # initialize tensorboard logger
        Configuration.tensorboard_folder = os.path.join(Configuration.output_directory, "tensorboard")
        if not os.path.exists(Configuration.tensorboard_folder):
            os.makedirs(Configuration.tensorboard_folder)
        self.writer = SummaryWriter(log_dir=Configuration.tensorboard_folder)

        # Print model summary
        cropped_image_size = Configuration.get("training.general.cropped_image_size")
        input_size = tuple(np.insert(cropped_image_size, 0, values=3))
        Logcreator.info(summary(self.model, input_size=input_size, device=DEVICE))

        Logcreator.debug("Model '%s' initialized with %d parameters." %
                         (Configuration.get('training.model.name'),
                          sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

    def plot_model(self):
        # TODO: This function should plot the model
        return 0

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
            Logcreator.info(f"Epoch {epoch}")

            train_loss, train_acc, train_patch_acc = self.train_step(train_loader, epoch)

            val_loss, val_acc, val_patch_acc = self.evaluate(self.model, val_loader, epoch)

            if self.swa_enabled and epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                # update batch normalization statistics for the swa_model
                torch.optim.swa_utils.update_bn(train_loader, self.swa_model, device=DEVICE)
                # evaluate on validation set
                swa_val_loss, swa_val_acc, swa_val_patch_acc = self.evaluate(self.swa_model, val_loader, epoch,
                                                                             log_model_name="SWA-",
                                                                             log_postfix_path='val_swa')

            # save model
            if (epoch % train_parms.checkpoint_save_interval == train_parms.checkpoint_save_interval - 1) or (
                    epoch + 1 == train_parms.num_epochs and DEVICE == "cuda"):
                self.save_model(epoch)
                self.save_checkpoint(self.model, epoch, train_loss, train_acc, val_loss, val_acc)

                if self.swa_enabled and epoch >= self.swa_start_epoch:
                    # update batch normalization statistics for the swa_model at the end
                    torch.optim.swa_utils.update_bn(train_loader, self.swa_model, device=DEVICE)
                    # save swa model separately
                    self.save_checkpoint(self.swa_model, epoch, train_loss, train_acc, swa_val_loss, swa_val_acc,
                                         file_name="swa_checkpoint.pth")

            epoch += 1

        # flush writer
        self.writer.flush()

        # TODO: Maybe save the images also in tensorbaord log (every other epoch?)
        # save predicted validation images
        save_predictions_as_imgs(val_loader, self.model,
                                 folder=os.path.join(Configuration.output_directory, "prediction-validation-set"),
                                 device=DEVICE,
                                 is_prob=False,
                                 pixel_threshold=Configuration.get("inference.general.foreground_threshold"))

        return 0

    def compute_loss(self, predictions, targets):
        if self.submission_loss:
            avgPool = torch.nn.AvgPool2d(16, stride=16)
            predictions = avgPool(predictions)
            targets = avgPool(targets)

        return self.loss_function(predictions, targets)

    def train_step(self, data_loader, epoch):
        """
        Train model for 1 epoch.
        """
        self.model.train()

        # initialize metrics
        multi_accuracy_metric = GeneralAccuracyMetric(device=DEVICE)

        accuracy = torchmetrics.Accuracy(threshold=Configuration.get('training.general.foreground_threshold'))
        accuracy.to(DEVICE)

        patch_accuracy = PatchAccuracy(threshold=Configuration.get('training.general.foreground_threshold'))
        patch_accuracy.to(DEVICE)

        total_loss = 0.

        # progressbar
        loop = tqdm(data_loader)

        # for all batches
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.float().unsqueeze(1).to(device=DEVICE)

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

            # update tqdm progressbar
            loop.set_postfix(loss=loss.item())

        self.lr_scheduler.step()  # decay learning rate over time

        # compute epoch scores
        train_loss = total_loss / len(data_loader)
        train_acc = accuracy.compute()
        train_patch_acc = patch_accuracy.compute()

        # log scores
        multi_accuracy_metric.compute_and_log(self.writer, epoch, path_postfix='train')

        self.writer.add_scalar("Loss/train", train_loss, epoch)
        self.writer.add_scalar("Accuracy/train", train_acc, epoch)
        self.writer.add_scalar("PatchAccuracy/train", train_patch_acc, epoch)

        Logcreator.info(f"Training:   loss: {train_loss:.5f}",
                        f", accuracy: {train_acc:.5f}",
                        f", patch-acc: {train_patch_acc:.5f}")

        return train_loss, train_acc, train_patch_acc

    def evaluate(self, model, data_loader, epoch, log_model_name='', log_postfix_path='val'):
        """
        Evaluate model on validation data.
        """
        model.eval()

        # initialize metrics
        multi_accuracy_metric = GeneralAccuracyMetric(device=DEVICE)

        accuracy = torchmetrics.Accuracy(threshold=Configuration.get('training.general.foreground_threshold'))
        accuracy.to(DEVICE)

        patch_accuracy = PatchAccuracy(threshold=Configuration.get('training.general.foreground_threshold'))
        patch_accuracy.to(DEVICE)

        total_loss = 0.

        with torch.no_grad():
            for i, (image, targets) in enumerate(data_loader):
                image, targets = image.to(DEVICE), targets.float().unsqueeze(1).to(DEVICE)

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

        # compute epoch scores
        val_loss = total_loss / len(data_loader)
        val_acc = accuracy.compute()
        val_patch_acc = patch_accuracy.compute()

        # log scores
        multi_accuracy_metric.compute_and_log(self.writer, epoch, path_postfix=log_postfix_path)

        self.writer.add_scalar("Loss/" + log_postfix_path, val_loss, epoch)
        self.writer.add_scalar("Accuracy/" + log_postfix_path, val_acc, epoch)
        self.writer.add_scalar("PatchAccuracy/" + log_postfix_path, val_patch_acc, epoch)

        Logcreator.info(log_model_name + f"Validation: loss: {val_loss:.5f}",
                        f", accuracy: {val_acc:.5f}",
                        f", patch-acc: {val_patch_acc:.5f}")

        return total_loss, val_acc, val_patch_acc

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

    def load_checkpints(self, path=None):
        checkpoint = torch.load(path)
        # Todo: check if to device should be called somewhere
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        train_accuracy = checkpoint['train_accuracy']
        val_loss = checkpoint['val_loss']
        val_accuracy = checkpoint['val_accuracy']
        # ToDo: add patch_accuracy to checkpoints

        return epoch, train_loss, train_accuracy, val_loss, val_accuracy
