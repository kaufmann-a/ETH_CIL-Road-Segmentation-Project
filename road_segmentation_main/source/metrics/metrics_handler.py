#!/usr/bin/env python3
# coding: utf8

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import torch
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric

from source.configuration import Configuration
from source.metrics.metrics import PatchAccuracy, PostProcessingPatchAccuracy, PostProcessingPixelAccuracy


class MetricsHandler(Metric):
    """
    Handles a group of metrics by updating, computing and logging them.
    """

    def __init__(self, device):
        """
        Initialize all metrics and place them on the given device.

        :param device: cpu or cuda or ...
        """
        super().__init__()

        self.metric_list, self.int_targets_list, self.names_list = self.get_metrics(device)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """
        Updates the internal state of all metrics.

        :param preds: The model predictions.
        :param target: The ground truth targets.
        """
        for metric, is_int_targets in zip(self.metric_list, self.int_targets_list):
            metric.update(preds, target.int() if is_int_targets else target)

    def compute(self):
        result_list = []
        for metric in self.metric_list:
            result_list.append(metric.compute().item())

        return result_list

    def compute_and_log(self, writer: SummaryWriter, comet, epoch, data_set_name='train'):
        """
        Computes and logs the metrics.

        :param writer: The tensorboard summary writer.
        :param comet: The comet handle.
        :param epoch: The current epoch number.
        :param data_set_name: The dataset name used for logging.
        :return: A score dictionary.
        """
        result_list = self.compute()

        score_dict = dict()
        for score, name in zip(result_list, self.names_list):
            # tensorboard
            writer.add_scalar(name + "/" + data_set_name, score, epoch)

            # dict
            score_name = data_set_name + "_" + name
            score_dict[score_name] = score

            # comet
            if comet is not None:
                comet.log_metric(score_name, score, epoch=epoch)

        return score_dict

    def get_metrics(self, device):
        """

        :param device: The used device.
        :return:
            metrics_list: A list containing the torchmetrics. <br>
            int_targets_list: A boolean list telling for each metric if the targets should be converted to integers.
            names_list: A list metric names.
        """
        metrics_list = []
        int_targets_list = []
        names_list = []

        foreground_threshold = Configuration.get('training.general.foreground_threshold')

        accuracy = torchmetrics.Accuracy(threshold=foreground_threshold).to(device)
        metrics_list.append(accuracy)
        int_targets_list.append(True)
        names_list.append("acc")

        patch_accuracy = PatchAccuracy(threshold=foreground_threshold).to(device)
        metrics_list.append(patch_accuracy)
        int_targets_list.append(False)
        names_list.append("patch_acc")

        iou = torchmetrics.IoU(num_classes=2, threshold=foreground_threshold).to(device)
        metrics_list.append(iou)
        int_targets_list.append(True)
        names_list.append("iou_score")

        morphparam = Configuration.get('training.postprocessing.morphology')

        postprocessingpatch_accuracy = PostProcessingPatchAccuracy(morphparam, device=device,
                                                                   threshold=foreground_threshold).to(device)
        metrics_list.append(postprocessingpatch_accuracy)
        int_targets_list.append(False)
        names_list.append("postprocessingpatch_acc")

        postprocessingpixel_accuracy = PostProcessingPixelAccuracy(morphparam, device=device,
                                                                   threshold=foreground_threshold).to(device)
        metrics_list.append(postprocessingpixel_accuracy)
        int_targets_list.append(False)
        names_list.append("postprocessingpixel_acc")

        return metrics_list, int_targets_list, names_list
