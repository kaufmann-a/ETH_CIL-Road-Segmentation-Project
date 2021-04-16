import torch
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