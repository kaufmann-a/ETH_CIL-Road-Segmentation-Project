import os

import torch
import torchvision


def save_predictions_as_imgs(loader, model, folder="../data/train-predictions", device="cuda", is_prob=True):
    """
    Save the predictions of the model as images.

    :param loader: the data loader to use
    :param model: the model to get the predictions
    :param folder: output folder to save the images in (is created if it does not exist)
    :param device: cuda/cpu
    :param is_prob: True = The output of the model are probabilities.
    :return:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)

    model.eval()

    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device).float().unsqueeze(1)

        with torch.no_grad():
            preds = model(x)
            if not is_prob:
                preds = torch.sigmoid(preds)

            # probabilities to 0/1
            preds = (preds >= 0.5).float()

            # save predictions
            torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")

            # save truth
            torchvision.utils.save_image(y, f"{folder}/pred_y_{idx}.png")

    model.train()
