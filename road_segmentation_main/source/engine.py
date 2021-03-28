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
from tqdm import tqdm

# TODO: from source.callbacks.callbacksfactory import CallbacksFactory
from source.configuration import Configuration
# from source.data.datagenerator import DataGenerator
from source.data.datapreparator import DataPreparator
from source.logcreator.logcreator import Logcreator
from source.helpers import converter
from source.lossfunctions.lossfunctionfactory import LossFunctionFactory
# TODO: from source.metrics.metricsfactory import MetricsFactory
from source.models.modelfactory import ModelFactory
from source.optimizers.optimizerfactory import OptimizerFactory
from source.helpers.utils import save_predictions_as_imgs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Engine:

    def __init__(self):
        self.model = ModelFactory.build().to(DEVICE)
        self.optimizer = OptimizerFactory.build(self.model)
        self.loss_function = LossFunctionFactory.build(self.model)
        self.scaler = torch.cuda.amp.GradScaler()  # I assumed we always use gradscaler, thus no factory for this

        Logcreator.debug("Model '%s' initialized with %d parameters." %
                         (Configuration.get('training.model.name'),
                          sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        # TODO: Print summary
        # Logcreator.debug(summary(self.model, (3, 400, 400))) Not working, check what parameters to input to the function

    def plot_model(self):
        # TODO: This function should plot the model
        return 0

    def train(self, epoch_start=0):
        training_data, validation_data, test_data = DataPreparator.load()

        batch_size = Configuration.get('training.batch_size')
        num_workers = Configuration.get('training.num_workers')
        shuffle = Configuration.get('training.shuffle')

        train_loader = DataLoader(training_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                  shuffle=shuffle)
        val_loader = DataLoader(validation_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                shuffle=False)  # TODO: check what shuffle exactly does and how to use it
        self.test_data = test_data

        num_epochs = Configuration.get('training.num_epochs')
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_step(train_loader)

            Logcreator.info(f"Epoch {epoch}")
            Logcreator.info(f"Training:   loss: {train_loss:.5f}", f", accuracy: {train_acc:.5f}")

            val_loss, val_acc = self.evaluate(val_loader)
            Logcreator.info(f"Validation: loss: {val_loss:.5f}", f", accuracy: {val_acc:.5f}")

            # save model
            checkpoint = {"state_dict": self.model.state_dict(), "optimizer": self.optimizer.state_dict()}
            # TODO: Implement save checkpionts
            # save_checkpoint(checkpoint)

        # TODO: Implement some examples to a folder
        # print some examples to a folder
        # save_predictions_as_imgs(val_loader, self.model, folder="./trainings", device=DEVICE, is_prob=False)

        return 0

    def train_step(self, data_loader):
        """
        Train model for 1 epoch.
        """
        self.model.train()

        # initialize metrics
        accuracy = torchmetrics.Accuracy(threshold=0.5)
        accuracy.to(DEVICE)

        total_loss = 0.

        # progressbar
        loop = tqdm(data_loader)

        # for all batches
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.float().unsqueeze(1).to(device=DEVICE)

            # forward
            with torch.cuda.amp.autocast():  # improve performance while maintaining accuracy
                predictions = self.model(data)
                loss = self.loss_function(predictions, targets)

            # backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # update loss
            total_loss += loss.item()

            # update metrics
            # TODO create one Metrics class that we can feed with (predicted, targets)
            #  and computes all metrics we want and maybe logs them -> tensorboard?
            accuracy.update(torch.sigmoid(predictions), targets.int())

            # update tqdm progressbar
            loop.set_postfix(loss=loss.item())

        total_loss /= len(data_loader)

        return total_loss, accuracy.compute()

    def evaluate(self, data_loader):
        """
        Evaluate model on validation data.
        """
        self.model.eval()

        # initialize metrics
        accuracy = torchmetrics.Accuracy()
        accuracy.to(DEVICE)

        total_loss = 0.

        with torch.no_grad():
            for i, (image, targets) in enumerate(data_loader):
                image, targets = image.to(DEVICE), targets.float().unsqueeze(1).to(DEVICE)

                # forward pass
                with torch.cuda.amp.autocast():
                    output = self.model(image)
                    loss = self.loss_function(output, targets)

                # update loss
                total_loss += loss.item()

                # update metrics
                accuracy.update(torch.sigmoid(output), targets.int())

        total_loss /= len(data_loader)

        return total_loss, accuracy.compute()

    def save(self):
        # TODO: Evaluate need of this function (should save model)
        # Do we need this??? does it exist in pytorch
        return 0

    def save_weights(self):
        # ToDo: Implement save_weights function

        # path = os.path.join(Configuration.get_path(
        #     'environment.weights.folder', optional=False), Configuration.get('environment.weights.file', optional=False))
        # self.model.save_weights(path, overwrite=True)
        # Logcreator.info("Saved weights to: %s" % path)
        return 0

    def load(self, path=None):
        # TODO: Do we need this? Is it available in pytorch? (load model from file)
        return 0

    def load_weights(self, path=None):
        # TODO: Implement function to load weight of an interrupted training

        # if not path:
        #     path = os.path.join(Configuration.get_path(
        #         'environment.weights.folder', optional=False), Configuration.get('environment.weights.file', optional=False))
        # Logcrator.info("Load weights from: %s" % path)
        # self.model.load_weights(path)
        return 0
