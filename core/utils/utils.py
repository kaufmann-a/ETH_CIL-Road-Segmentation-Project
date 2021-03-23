import os

import torch
import torchvision


def save_predictions_as_imgs(loader, model, folder="../data/train-predictions", device="cuda"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device).float().unsqueeze(1)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )

            torchvision.utils.save_image(y, f"{folder}/pred_y_{idx}.png")

    model.train()
