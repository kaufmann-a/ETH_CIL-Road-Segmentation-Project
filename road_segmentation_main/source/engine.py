#!/usr/bin/env python3
# coding: utf8

"""
Engine of the road segmentation neuronal network learning object.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

# noinspection PyUnresolvedReferences
from comet_ml import Experiment  # comet_ml needs to be imported before torch

import sys
import os
import numpy as np
import random
import torch
import inspect

from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from torchsummary import summary
from tqdm import tqdm
from io import StringIO

from source.configuration import Configuration
from source.data.datapreparator import DataPreparator
from source.helpers.imagesavehelper import save_predictions_as_imgs
import source.helpers.metricslogging as metricslogging
from source.logcreator.logcreator import Logcreator
from source.lossfunctions.lossfunctionfactory import LossFunctionFactory
from source.metrics.metrics_handler import MetricsHandler
from source.models.modelfactory import ModelFactory
from source.optimizers.optimizerfactory import OptimizerFactory
from source.scheduler.lr_schedulerfactory import LRSchedulerFactory
import source.helpers.imagesavehelper as imagesavehelper

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Engine:
    """
    Handles the training of the model.
    """

    def __init__(self, args=None):
        """
        Initializes the model, optimizer, learning rate scheduler and the loss function
        using the respective Factory classes.

        :param args:
        """

        # fix random seeds
        seed = 49626446
        self.fix_random_seeds(seed)
        self.fix_deterministic_operations()
        if args is not None and args.lines_layer_path != '':
            self.lines_layer_path = args.lines_layer_path
            self.predicted_masks_path = args.predicted_masks_path

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

        # Print model summary
        self.print_modelsummary()

        if inspect.getouterframes(inspect.currentframe(), 2)[1].filename.__contains__('train.py'):
            # Init comet and tensorboard
            self.comet = metricslogging.init_comet()
            self.tensorboard = metricslogging.init_tensorboard()

        # fix random seeds again after initialisation, such that changing the model
        # does not influence the seeds of the data loading, training, ...
        self.fix_random_seeds(seed=35416879)

    def get_lr(self):
        """
        :return: the current learning rate
        """
        return self.optimizer.param_groups[0]['lr']

    def train(self, epoch_nr=0):
        """
        This is the main training loop function. It trains, validates and saves the best model for the given number of
        epochs.

        :param epoch_nr: Start training from this epoch. Useful to continue training with a saved model checkpoint.
        """
        training_data, validation_data = DataPreparator.load(self)

        # Load training parameters from config file
        train_parms = Configuration.get('training.general')

        # drop the last batch if it only has size one, because otherwise an error is thrown
        size_of_last_batch = len(training_data) % train_parms.batch_size
        drop_last_incomplete_batch_train = size_of_last_batch == 1
        if drop_last_incomplete_batch_train:
            Logcreator.info("Training - Last batch dropped of size", size_of_last_batch)

        # set random number generator for the train dataloader explicitly
        g = torch.Generator()
        g.manual_seed(16871643)

        train_loader = DataLoader(training_data, batch_size=train_parms.batch_size, num_workers=train_parms.num_workers,
                                  pin_memory=True,
                                  worker_init_fn=seed_worker,
                                  generator=g,
                                  shuffle=train_parms.shuffle_data,
                                  drop_last=drop_last_incomplete_batch_train)

        val_loader = DataLoader(validation_data, batch_size=train_parms.batch_size, num_workers=train_parms.num_workers,
                                worker_init_fn=seed_worker,
                                pin_memory=True,
                                shuffle=False)

        epoch = 0
        if epoch_nr != 0:  # Check if continued training
            epoch = epoch_nr + 1  # plus one to continue with the next epoch

        nr_saves = 0
        best_val_accuracy = 0.0
        while epoch < train_parms.num_epochs:
            Logcreator.info(f"Epoch {epoch}, lr: {self.get_lr():.3e}, lr-step: {self.lr_scheduler.last_epoch}")

            train_metrics = self.train_step(train_loader, epoch)
            val_metrics = self.evaluate(self.model, val_loader, epoch)

            # swa model
            if self.swa_enabled and epoch >= self.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
                # update batch normalization statistics for the swa_model
                with torch.no_grad():
                    torch.optim.swa_utils.update_bn(tqdm(train_loader, desc="SWA BN update", file=sys.stdout),
                                                    self.swa_model, device=DEVICE)
                # evaluate on validation set
                swa_val_metrics = self.evaluate(self.swa_model, val_loader, epoch, log_dataset_name='val_swa')

            best_model = False
            if val_metrics['val_acc'] > best_val_accuracy:
                best_val_accuracy = val_metrics['val_acc']
                Logcreator.info("New best model with validation acc", best_val_accuracy)
                best_model = True

            # save model
            if (epoch % train_parms.checkpoint_save_interval == train_parms.checkpoint_save_interval - 1) \
                    or (epoch + 1 == train_parms.num_epochs and DEVICE == "cuda") \
                    or (best_model and epoch > 9):
                # self.save_model(epoch)
                self.save_checkpoint(epoch,
                                     train_metrics['train_loss'], train_metrics['train_acc'],
                                     val_metrics['val_loss'], val_metrics['val_acc'],
                                     file_name="best.pth" if best_model else "checkpoint.pth")

            if self.comet is not None:
                imagesavehelper.save_predictions_to_comet(self,
                                                          val_loader,
                                                          epoch,
                                                          Configuration.get("inference.general.foreground_threshold"),
                                                          DEVICE, False,
                                                          nr_saves)
                nr_saves += 1

            # Flush tensorboard
            self.tensorboard.flush()

            epoch += 1

        # save predicted validation images
        save_imgs = Configuration.get("data_collection.save_imgs")
        if save_imgs:
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
        metrics_handler = MetricsHandler(device=DEVICE)

        total_loss = 0.

        # progressbar
        loop = tqdm(data_loader, file=sys.stdout, colour='green')

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
            prediction_probabilities = torch.sigmoid(predictions)
            metrics_handler.update(prediction_probabilities, targets)

            # update tqdm progressbar
            loop.set_postfix(loss=loss.item())

        self.lr_scheduler.step()  # decay learning rate over time

        # compute epoch scores
        train_loss = total_loss / len(data_loader)

        # Tensorboard
        self.tensorboard.add_scalar("Loss/train", train_loss, epoch)

        # Comet
        if self.comet is not None:
            self.comet.log_metric('train_loss', train_loss, epoch=epoch)

        score_dict = metrics_handler.compute_and_log(self.tensorboard, self.comet, epoch=epoch, data_set_name="train")

        Logcreator.info(f"Training:   loss: {train_loss:.5f}", self.score_dict_to_string(score_dict))

        # add the loss to the dictionary
        score_dict["train_loss"] = train_loss

        return score_dict

    def evaluate(self, model, data_loader, epoch, log_dataset_name='val'):
        """
        Evaluate model on validation data.
        """
        model.eval()

        # initialize metrics
        metrics_handler = MetricsHandler(device=DEVICE)

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
                metrics_handler.update(prediction_probabilities, targets)

        # compute epoch scores
        val_loss = total_loss / len(data_loader)

        # Tensorboard
        self.tensorboard.add_scalar("Loss/" + log_dataset_name, val_loss, epoch)

        # Comet
        if self.comet is not None:
            self.comet.log_metric(log_dataset_name + '_loss', val_loss, epoch=epoch)

        score_dict = metrics_handler.compute_and_log(self.tensorboard, self.comet, epoch=epoch,
                                                     data_set_name=log_dataset_name)

        Logcreator.info(f"Validation: loss: {val_loss:.5f}", self.score_dict_to_string(score_dict))

        # add the loss to the dictionary
        score_dict["val_loss"] = val_loss

        return score_dict

    def compute_loss(self, predictions, targets):
        """
        Computes the loss using the configured loss function. If submission-loss is configured the prediction and target
        masks are first converted to the submission mask format. The submission mask format only takes 16x16 patches
        into account.

        :param predictions: The model predictions.
        :param targets: The ground truth mask.

        :return: The computed loss.
        """
        if self.submission_loss:
            # use average pooling to convert the predictions and targets to the submission format
            avgPool = torch.nn.AvgPool2d(16, stride=16)
            predictions = avgPool(predictions)
            targets = avgPool(targets)

        return self.loss_function(predictions, targets)

    def score_dict_to_string(self, score_dict):
        """
        Helper to convert the scores dictionary to a string.

        :param score_dict: a dictionary containing float values

        :return: the string representing the dictionary
        """
        return ''.join(f'{name.replace("val_", "").replace("train_", "")}: {score:.5f}, '
                       for name, score in score_dict.items())

    def save_model(self, epoch_nr):
        """
        This function saves the entire model incl. model structure.

        :param epoch_nr: The current epoch number.
        """
        Configuration.model_save_folder = os.path.join(Configuration.output_directory, "whole_model_backups")
        if not os.path.exists(Configuration.model_save_folder):
            os.makedirs(Configuration.model_save_folder)
        file_name = str(epoch_nr) + "_whole_model_serialized.pth"
        torch.save(self.model, os.path.join(Configuration.model_save_folder, file_name))

    def save_checkpoint(self, epoch, tl, ta, vl, va, file_name="checkpoint.pth"):
        """
        Saves a model checkpoint.

        :param epoch: The current epoch number.
        :param tl: training loss
        :param ta: training accuracy
        :param vl: validation loss
        :param va: validation accuracy
        :param file_name: The checkpoint file name.
        """
        Configuration.weights_save_folder = os.path.join(Configuration.output_directory, "weights_checkpoint")
        if not os.path.exists(Configuration.weights_save_folder):
            os.makedirs(Configuration.weights_save_folder)
        file_name = str(epoch) + "_" + file_name

        if self.swa_enabled and epoch >= self.swa_start_epoch:
            swa_model_state_dict = self.swa_model.state_dict()
        else:
            swa_model_state_dict = None

        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'swa_model_state_dict': swa_model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': tl,
            'train_accuracy': ta,
            'val_loss': vl,
            'val_accuracy': va,
        }, os.path.join(Configuration.weights_save_folder, file_name))

    def load_model(self, path=None):
        """
        Loads a saved model.

        :param path: Path to the saved model structure.
        """
        self.model = torch.load(path)

    def load_checkpoints(self, path=None, reset_lr=False, overwrite_model_with_swa=False):
        """
        Loads a model checkpoint.

        :param path: The path to checkpoint file.
        :param reset_lr: True = resets the learning rate of the optimizer to the configuration values.
        :param overwrite_model_with_swa: True = Overwrites the model with the saved swa model.

        :return: epoch number, train loss, train accuracy, validation loss, validation accuracy
        """
        Logcreator.info("Loading checkpoint file:", path)

        checkpoint = torch.load(path)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        # load swa model if saved in checkpoint
        if 'swa_model_state_dict' in checkpoint and checkpoint['swa_model_state_dict'] is not None:
            # Load the swa model
            self.swa_model.load_state_dict(checkpoint['swa_model_state_dict'])
        else:
            self.swa_model = AveragedModel(self.model)

        if overwrite_model_with_swa:
            # Overwrite the model with the swa model
            self.model.load_state_dict(self.swa_model.module.state_dict())

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

    def fix_random_seeds(self, seed):
        """
        Fixes the random seeds of torch, numpy, random and cuda.

        :param seed: The seed number.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # this is probably overkill, since these functions are also called in torch.manual_seed
        #  (but there is a if condition in there, that then might prevent to call these)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)  # current gpu
            torch.cuda.manual_seed_all(seed)  # all gpus

    def fix_deterministic_operations(self):
        """
        Fixes backend cudnn to produce deterministic results.

        """
        if torch.cuda.is_available():
            benchmark = Configuration.get("training.cudnn.benchmark", optional=True, default=False)
            deterministic = Configuration.get("training.cudnn.deterministic", optional=True, default=True)
            Logcreator.info("cudnn.benchmark default:", torch.backends.cudnn.benchmark, ", set-value:", benchmark)
            Logcreator.info("cudnn.deterministic default:", torch.backends.cudnn.deterministic, ", set-value:",
                            deterministic)
            torch.backends.cudnn.benchmark = benchmark
            torch.backends.cudnn.deterministic = deterministic  # if True: CUDA convolution deterministic
            if deterministic:  # set workspace config, to maybe give same results on 1080Ti and 2080Ti
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
            # torch.use_deterministic_algorithms(True) # last resort if other things do not work

    def print_modelsummary(self):
        """
        Logs the model summary.

        """
        nr_channels = 3
        if hasattr(self, 'lines_layer_path'):
            nr_channels = 5
            Logcreator.info("Lines layer path :", self.lines_layer_path)
            Logcreator.info("Predicted layer path :", self.predicted_masks_path)

        cropped_image_size = Configuration.get("training.general.cropped_image_size")
        input_size = tuple(np.insert(cropped_image_size, 0, values=nr_channels))
        # redirect stdout to our logger
        sys.stdout = mystdout = StringIO()
        summary(self.model, input_size=input_size, device=DEVICE)
        # reset stdout to original
        sys.stdout = sys.__stdout__
        Logcreator.info(mystdout.getvalue())

        Logcreator.debug("Model '%s' initialized with %d parameters." %
                         (Configuration.get('training.model.name'),
                          sum(p.numel() for p in self.model.parameters() if p.requires_grad)))


def seed_worker(worker_id):
    """
    Fixes seeds for numpy and random.

    :param worker_id: The worker id.
    """
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    # print("worker-id:", worker_id, ", seed:", worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
