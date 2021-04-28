import os

import torch
import torchvision


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


def save_predictions_as_imgs(loader, model, folder="../data/train-predictions", pixel_threshold=0.5, device="cuda",
                             is_prob=False,
                             individual_saving=True):
    """
    Save the predictions of the model as images.

    :param loader: the data loader to use
    :param model: the model to get the predictions
    :param folder: output folder to save the images in (is created if it does not exist)
    :param pixel_threshold: probability threshold that a pixel is a road
    :param device: cuda/cpu
    :param is_prob: True = The output of the model are probabilities.
    :individual_saving: True = Every predicted mask is saved separately to a file.
    :return:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    model.eval()

    pred_img_idx = 0

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device).float().unsqueeze(1)

        with torch.no_grad():
            preds = model(x)
            if not is_prob:
                preds = torch.sigmoid(preds)

            # probabilities to 0/1
            preds = (preds > pixel_threshold).float()

            if individual_saving:
                # save every prediction separately
                for i in range(0, preds.shape[0]):
                    # save prediction
                    torchvision.utils.save_image(preds[i], f"{folder}/pred_{pred_img_idx}.png")
                    # save truth
                    torchvision.utils.save_image(y[i], f"{folder}/true_{pred_img_idx}.png")
                    # save input
                    torchvision.utils.save_image(x[i], f"{folder}/input_{pred_img_idx}.png")

                    pred_img_idx += 1
            else:
                # save entire batch predictions as one grid image
                torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
                # save truth
                torchvision.utils.save_image(y, f"{folder}/true_{idx}.png")
                # save input
                torchvision.utils.save_image(x, f"{folder}/input_{idx}.png")

    model.train()


def save_masks_as_images(preds, index, folder, pixel_threshold=0.5, is_prob=True, save_submission_img=True):
    """
    Save the predictions of the model as images.

    :preds: list of predictions
    :index: the index list of the individual predictions
    :param folder: output folder to save the images in (is created if it does not exist)
    :param pixel_threshold: probability threshold that a pixel is a road
    :param is_prob: True = The output of the model are probabilities.
    :return:

    """
    folder_normal_size = os.path.join(folder, "pred-masks-original")
    if not os.path.exists(folder_normal_size):
        os.makedirs(folder_normal_size)

    if save_submission_img:
        folder_small_size = os.path.join(folder, "pred-mask-submission")
        if not os.path.exists(folder_small_size):
            os.makedirs(folder_small_size)

    if not is_prob:
        preds = torch.sigmoid(preds)

    for i in range(len(preds)):

        # probabilities to 0/1
        out_preds = (preds[i] > pixel_threshold).float()
        # save prediction
        torchvision.utils.save_image(out_preds, f"{folder_normal_size}/pred_{index[i]}.png")

        if save_submission_img:
            # save submission masks
            patched_preds = mask_to_submission_mask(torch.unsqueeze(preds[i], 0), pixel_threshold).float()
            # save prediction
            torchvision.utils.save_image(patched_preds, f"{folder_small_size}/pred_{index[i]}.png")
