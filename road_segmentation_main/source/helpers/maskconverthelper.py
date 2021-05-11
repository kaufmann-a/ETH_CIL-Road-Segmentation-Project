
import torch



def mask_to_submission_mask(mask, threshold):
    """
    Converts a mask of original size to a submission mask.
    One pixel in the submission mask corresponds to a 16x16 patch of the original mask.

    :param mask: original mask
    :param threshold: probability threshold
    :return:
    """
    # save submission masks
    avgPool = torch.nn.AvgPool2d(16, stride=16)
    submission_mask = avgPool(mask)

    # convert to integers according to threshold
    submission_mask = (submission_mask > threshold).int()
    return submission_mask