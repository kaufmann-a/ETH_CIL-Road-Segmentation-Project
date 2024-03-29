"""
Helps saving model predictions.
"""

__author__ = 'Andreas Kaufmann, Jona Braun, Frederike Lübeck, Akanksha Baranwal'
__email__ = "ankaufmann@student.ethz.ch, jonbraun@student.ethz.ch, fluebeck@student.ethz.ch, abaranwal@student.ethz.ch"

import os
import torch
import torchvision as torchvision

from source.helpers.maskconverthelper import mask_to_submission_mask


def save_predictions_to_comet(engine, loader, epoch, pixel_threshold, device, is_prob, nr_saves):
    """
    Uploads the predictions of the given model using the provided data loader to comet.

    :param engine: The main engine.
    :param loader: A pytroch data loader.
    :param epoch: The current epoch number which is used for the naming of the predictions.
    :param pixel_threshold: The probabilistic threshold that decides that a predicted pixel is a road pixel.
    :param device: cpu or cuda
    :param is_prob: True = The output of the model is probabilistic.
    :param nr_saves: External counter variable to indicate the number of uploads done so far.
    :return:
    """
    engine.model.eval()

    pred_img_idx = 0

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device).float()

        with torch.no_grad():
            preds = engine.model(x)
            if not is_prob:
                preds = torch.sigmoid(preds)
            # probabilities to 0/1
            preds = (preds > pixel_threshold).float()
            # save every prediction separately
            for i in range(0, preds.shape[0]):
                if i % 10 == 0: # Just every 10th image is saved
                    with engine.comet.context_manager(f"img_nr_{pred_img_idx}"):
                        if nr_saves == 0:
                            engine.comet.log_image(torchvision.transforms.ToPILImage()(x[i][:3,:,:]),
                                                   f"{pred_img_idx}_1_input.png", image_format="png")
                            engine.comet.log_image(torchvision.transforms.ToPILImage()(y[i]),
                                                   f"{pred_img_idx}_2_true.png", image_format="png")
                        engine.comet.log_image(torchvision.transforms.ToPILImage()(preds[i]),
                                               f"{pred_img_idx}_3_pred_epoch_{epoch:03d}.png", image_format="png")
                pred_img_idx += 1

    engine.model.train()


def save_predictions_as_imgs(loader, model, folder="../data/train-predictions", pixel_threshold=0.5, device="cuda",
                             is_prob=False,
                             individual_saving=True):
    """
    Save the predictions of the model as images.

    :param loader: A pytroch data loader.
    :param model: The model to use.
    :param folder: Output folder to save the images in (is created if it does not exist).
    :param pixel_threshold: Probability threshold that a pixel is a road.
    :param device: cuda/cpu
    :param is_prob: True = The output of the model are probabilities.
    :param individual_saving: True = Every predicted mask is saved separately to a file.
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
                    torchvision.utils.save_image(preds[i], f"{folder}/pred_{pred_img_idx}.jpg")
                    # save truth
                    torchvision.utils.save_image(y[i], f"{folder}/true_{pred_img_idx}.jpg")
                    # save input
                    torchvision.utils.save_image(x[i], f"{folder}/input_{pred_img_idx}.jpg")

                    pred_img_idx += 1
            else:
                # save entire batch predictions as one grid image
                torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.jpg")
                # save truth
                torchvision.utils.save_image(y, f"{folder}/true_{idx}.jpg")
                # save input
                torchvision.utils.save_image(x, f"{folder}/input_{idx}.jpg")

    model.train()


def save_masks_as_images(preds, index, folder, pixel_threshold=0.5, is_prob=True, save_submission_img=True,
                         folder_prefix=''):
    """
    Save the predictions of the model as images.

    :param preds: List of predictions.
    :param index: The index list of the individual predictions.
    :param folder: Output folder to save the images in (is created if it does not exist).
    :param pixel_threshold: Probability threshold that a pixel is a road.
    :param is_prob: True = The output of the model are probabilities.
    :param save_submission_img: Saves the submission image to a file.
    :param folder_prefix: Prefix for the folder where the images are saved.
    :return: predictions as masks
    """
    folder_normal_size = os.path.join(folder, folder_prefix + "pred-masks-original")
    if not os.path.exists(folder_normal_size):
        os.makedirs(folder_normal_size)

    if save_submission_img:
        folder_small_size = os.path.join(folder, folder_prefix + "pred-mask-submission")
        if not os.path.exists(folder_small_size):
            os.makedirs(folder_small_size)

    if not is_prob:
        preds = torch.sigmoid(preds)

    out_preds_list = []
    for i in range(len(preds)):

        # probabilities to 0/1
        out_preds = (preds[i] > pixel_threshold).float()
        out_preds_list.append(out_preds)
        # save prediction
        torchvision.utils.save_image(out_preds, f"{folder_normal_size}/pred_{index[i]}.png")

        if save_submission_img:
            # save submission masks
            patched_preds = mask_to_submission_mask(torch.unsqueeze(preds[i], 0), pixel_threshold).float()
            # save prediction
            torchvision.utils.save_image(patched_preds, f"{folder_small_size}/pred_{index[i]}.png")

    return out_preds_list
