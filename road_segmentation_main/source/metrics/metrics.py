import torch
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Metric


class PatchAccuracy(Metric):
    def __init__(self, threshold=0.5, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.threshold = threshold

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # create patches
        avgPool = torch.nn.AvgPool2d(16, stride=16)
        patched_preds = avgPool(preds)
        patched_target = avgPool(target)

        # convert to integers according to threshold
        patched_preds = (patched_preds > self.threshold).int()
        patched_target = (patched_target > self.threshold).int()

        # update metric states
        self.correct += torch.sum(patched_preds == patched_target)
        self.total += patched_target.numel()

    def compute(self):
        # compute final result
        return self.correct.float() / self.total


class GeneralAccuracyMetric(Metric):

    def __init__(self, device, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.thresholds = [0.25, 0.5, 0.75, 0.9]

        self.accuracy_list = list()
        self.patch_accuracy_list = list()

        for t in self.thresholds:
            self.accuracy_list.append(torchmetrics.Accuracy(threshold=t).to(device))
            self.patch_accuracy_list.append(PatchAccuracy(threshold=t).to(device))

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        for acc in self.accuracy_list:
            acc.update(preds, target.int())

        for p_acc in self.patch_accuracy_list:
            p_acc.update(preds, target)

        pass

    def compute(self):
        acc_out_list = list()
        p_acc_out_list = list()
        for acc in self.accuracy_list:
            acc_out_list.append(acc.compute())

        for acc in self.patch_accuracy_list:
            p_acc_out_list.append(acc.compute())

        return acc_out_list, p_acc_out_list

    def compute_and_log(self, writer: SummaryWriter, epoch, path_postfix='train'):
        acc_out_list, p_acc_out_list = self.compute()

        for t, i in zip(self.thresholds, range(len(self.thresholds))):
            writer.add_scalar('Accuracy/' + str(t) + "/" + path_postfix, acc_out_list[i], epoch)
            writer.add_scalar('PatchAccuracy/' + str(t) + "/" + path_postfix, acc_out_list[i], epoch)
