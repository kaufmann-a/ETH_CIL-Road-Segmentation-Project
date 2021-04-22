#!/usr/bin/env python3
# coding: utf8

"""
Model of the road segmentatjion neuronal network learning object.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import os

import torch
import torchmetrics as torchmetrics
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from tqdm import tqdm

# TODO: from source.callbacks.callbacksfactory import CallbacksFactory
from source.configuration import Configuration
# from source.data.datagenerator import DataGenerator
from source.data.datapreparator import DataPreparator
from source.helpers.utils import save_predictions_as_imgs
from source.logcreator.logcreator import Logcreator
from source.lossfunctions.lossfunctionfactory import LossFunctionFactory
# TODO: from source.metrics.metricsfactory import MetricsFactory
from source.metrics.metrics import PatchAccuracy
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

        # initialize tensorboard logger
        self.writer = SummaryWriter(log_dir=Configuration.output_directory)

        # Print model summary
        Logcreator.info(summary(self.model, input_size=(3, 400, 400), device=DEVICE))

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
            epoch = epoch_nr

        while epoch < train_parms.num_epochs:
            train_loss, train_acc, train_patch_acc = self.train_step(train_loader)

            Logcreator.info(f"Epoch {epoch}")
            Logcreator.info(f"Training:   loss: {train_loss:.5f}",
                            f", accuracy: {train_acc:.5f}",
                            f", patch-acc: {train_patch_acc:.5f}")

            val_loss, val_acc, val_patch_acc = self.evaluate(val_loader)
            Logcreator.info(f"Validation: loss: {val_loss:.5f}",
                            f", accuracy: {val_acc:.5f}",
                            f", patch-acc: {val_patch_acc:.5f}")

            # log scores
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Accuracy/train", train_acc, epoch)
            self.writer.add_scalar("PatchAccuracy/train", train_patch_acc, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Accuracy/val", val_acc, epoch)
            self.writer.add_scalar("PatchAccuracy/val", train_patch_acc, epoch)

            # save model
            if epoch % train_parms.checkpoint_save_interval == train_parms.checkpoint_save_interval - 1:
                self.save_model(epoch)
                self.save_checkpoint(epoch, train_loss, train_acc, val_loss, val_acc)

            epoch += 1

        # flush writer
        self.writer.flush()

        # TODO: Maybe save the images also in tensorbaord log (every other epoch?)
        # save predicted validation images
        save_predictions_as_imgs(val_loader, self.model,
                                 folder=os.path.join(Configuration.output_directory, "prediction"),
                                 device=DEVICE,
                                 is_prob=False,
                                 pixel_threshold=Configuration.get("data_collection.foreground_threshold"))

        return 0

    def train_step(self, data_loader):
        """
        Train model for 1 epoch.
        """
        self.model.train()


        # initialize metrics
        accuracy = torchmetrics.Accuracy(threshold=Configuration.get('data_collection.foreground_threshold'))
        accuracy.to(DEVICE)

        patch_accuracy = PatchAccuracy(threshold=Configuration.get('data_collection.foreground_threshold'))
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
                loss = self.loss_function(predictions, targets)

            # backward according to https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # update loss
            total_loss += loss.item()

            # update metrics
            # TODO create one Metrics class that we can feed with (predicted, targets)
            #  and computes all metrics we want and maybe logs them -> tensorboard?
            accuracy.update(torch.sigmoid(predictions), targets.int())
            patch_accuracy.update(torch.sigmoid(predictions), targets)

            # update tqdm progressbar
            loop.set_postfix(loss=loss.item())

        total_loss /= len(data_loader)

        self.lr_scheduler.step()  # decay learning rate over time

        return total_loss, accuracy.compute(), patch_accuracy.compute()

    def evaluate(self, data_loader):
        """
        Evaluate model on validation data.
        """
        self.model.eval()

        # initialize metrics
        accuracy = torchmetrics.Accuracy()
        accuracy.to(DEVICE)

        patch_accuracy = PatchAccuracy(threshold=Configuration.get('data_collection.foreground_threshold'))
        patch_accuracy.to(DEVICE)

        total_loss = 0.

        with torch.no_grad():
            for i, (image, targets) in enumerate(data_loader):
                image, targets = image.to(DEVICE), targets.float().unsqueeze(1).to(DEVICE)

                # forward pass according to https://pytorch.org/docs/stable/amp.html
                with torch.cuda.amp.autocast():
                    output = self.model(image)
                    loss = self.loss_function(output, targets)

                # update loss
                total_loss += loss.item()

                # update metrics
                accuracy.update(torch.sigmoid(output), targets.int())
                patch_accuracy.update(torch.sigmoid(output), targets)

        total_loss /= len(data_loader)

        return total_loss, accuracy.compute(), patch_accuracy.compute()

    def save_model(self, epoch_nr):
        """ This function saves entire model incl. modelstructure"""
        Configuration.model_save_folder = os.path.join(Configuration.output_directory, "whole_model_backups")
        if not os.path.exists(Configuration.model_save_folder):
            os.makedirs(Configuration.model_save_folder)
        file_name = str(epoch_nr) + "_whole_model_serialized.pth"
        torch.save(self.model, os.path.join(Configuration.model_save_folder, file_name))

    def save_checkpoint(self, epoch, tl, ta, vl, va):
        Configuration.weights_save_folder = os.path.join(Configuration.output_directory, "weights_checkpoint")
        if not os.path.exists(Configuration.weights_save_folder):
            os.makedirs(Configuration.weights_save_folder)
        file_name = str(epoch) + "_checkpoint.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
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
