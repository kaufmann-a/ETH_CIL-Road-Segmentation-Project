import os

import torch
import torchvision
import cv2
from source.postprocessing.postprocessing import postprocess
from tqdm import tqdm


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


def patch_to_label(foreground_threshold, patch):
    # assign a label to a patch
    df = torch.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings( image, image_nr, patch_size=16, foreground_threshold=0.25):
    # iterate over prediction, just use every 16th pixel
    for j in range(0, image.shape[1], patch_size):
        for i in range(0, image.shape[0], patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(foreground_threshold, patch)
            yield ("{:03d}_{}_{},{}".format(image_nr, j, i, label))


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

    out_preds_list =[]
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

# TODO: Handle if morphological operations aren't defined.
def runpostprocessing(preds_list, folder, postprocessingparams, image_number_list, patch_size, foreground_threshold):
    folder_postprocessed = os.path.join(folder, "pred-masks-postprocessed")
    if not os.path.exists(folder_postprocessed):
        os.makedirs(folder_postprocessed)

    with open(os.path.join(folder, 'postprocessed_submission.csv'), 'w') as f:
        f.write('id,prediction\n')
        image_nr_list_idx = 0

        for i in range(len(preds_list)):
            preds = preds_list[i]
            preds = preds.cpu().numpy()
            preds = preds.astype('uint8')
            postprocessed_img = postprocess(preds, postprocessingparams.morphology)
            cv2.imwrite(f"{folder_postprocessed}/pred_{image_number_list[i]}.png", 255*postprocessed_img)
            postprocessed_img = torch.tensor(postprocessed_img)
            postprocessed_img = postprocessed_img.to(torch.double)
            f.writelines('{}\n'.format(s)
                         for s in mask_to_submission_strings(image=postprocessed_img,
                                                             patch_size=patch_size,
                                                             image_nr=image_number_list[image_nr_list_idx],
                                                             foreground_threshold=foreground_threshold))
        image_nr_list_idx += 1
