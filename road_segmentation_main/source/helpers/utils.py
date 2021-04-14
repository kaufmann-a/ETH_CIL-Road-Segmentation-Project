import os

import torch
import torchvision


def save_predictions_as_imgs(loader, model, folder="../data/train-predictions", device="cuda",
                             is_prob=False,
                             individual_saving=False):
    """
    Save the predictions of the model as images.

    :param loader: the data loader to use
    :param model: the model to get the predictions
    :param folder: output folder to save the images in (is created if it does not exist)
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
            preds = (preds >= 0.5).float()

            if individual_saving:
                # save every prediction separately
                for i in range(0, preds.shape[0]):
                    torchvision.utils.save_image(preds[i], f"{folder}/pred_{pred_img_idx}.png")
                    pred_img_idx += 1
            else:
                # save entire batch predictions as one grid image
                torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")

            # save truth
            torchvision.utils.save_image(y, f"{folder}/true_{idx}.png")

            # save input
            torchvision.utils.save_image(x, f"{folder}/input_{idx}.png")

    model.train()
