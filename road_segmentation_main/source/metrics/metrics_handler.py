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

        :param device: cpu or cuda or ...
        """
        super().__init__()

        self.metric_list, self.int_targets_list = self.get_metrics(device)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        for metric, is_int_targets in zip(self.metric_list, self.int_targets_list):
            metric.update(preds, target.int() if is_int_targets else target)

    def compute(self):
        result_list = []
        for metric in self.metric_list:
            result_list.append(metric.compute().item())

        return result_list

    def compute_and_log(self, writer: SummaryWriter, comet, epoch, path_postfix='train'):
        result_list = self.compute()

        return result_list

    def get_metrics(self, device):
        """

        :param device: The used device.
        :return:
            metrics_list: A list containing the torchmetrics. <br>
            int_targets_list: A boolean list telling for each metric if the targets should be converted to integers.
        """
        metrics_list = []
        int_targets_list = []

        foreground_threshold = Configuration.get('training.general.foreground_threshold')

        accuracy = torchmetrics.Accuracy(threshold=foreground_threshold).to(device)
        metrics_list += [accuracy]
        int_targets_list += [True]

        iou = torchmetrics.IoU(num_classes=2, threshold=foreground_threshold).to(device)
        metrics_list += [iou]
        int_targets_list += [True]

        morphparam = Configuration.get('training.postprocessing.morphology')

        patch_accuracy = PatchAccuracy(threshold=foreground_threshold).to(device)
        metrics_list += [patch_accuracy]
        int_targets_list += [False]

        postprocessingpatch_accuracy = PostProcessingPatchAccuracy(morphparam, device=device,
                                                                   threshold=foreground_threshold).to(device)
        metrics_list += [postprocessingpatch_accuracy]
        int_targets_list += [False]

        postprocessingpixel_accuracy = PostProcessingPixelAccuracy(morphparam, device=device,
                                                                   threshold=foreground_threshold).to(device)
        metrics_list += [postprocessingpixel_accuracy]
        int_targets_list += [False]

        return metrics_list, int_targets_list
